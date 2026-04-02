"""
Depth-image student model for DAgger distillation (pure PyTorch, no rl_games).

Architecture
------------
DepthCNN          : (N, 1, H, W) → (N, 128, h, w)  [4-layer CNN + LayerNorm]
DepthTransformer  : spatial tokens → CLS token → (N, CNN_OUT)
concat with proprio → LSTM → MLP → mu / sigma / value

Interface (compatible with distillation.py)
------------------------------------------
forward(batch_dict) → {"mus", "sigmas", "values", "rnn_states"}
is_rnn()
get_default_rnn_state()
a2c_network  (property → self)
is_aux = False
running_mean_std  (simple running-stats module for depth)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CNN_OUT = 32        # depth feature dim fed into the MLP
IMG_H   = 240       # default depth image height  (120 * 2)
IMG_W   = 320       # default depth image width   (160 * 2)


# ──────────────────────────────────────────────────────────────────────────────
# Running mean / std  (no rl_games dependency)
# ──────────────────────────────────────────────────────────────────────────────

class RunningMeanStd(nn.Module):
    """Welford online running mean and variance."""

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean",  torch.zeros(shape))
        self.register_buffer("var",   torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))

    def forward(self, x):
        if self.training:
            b = x.shape[0]
            batch_mean = x.mean(dim=0)
            batch_var  = x.var(dim=0, unbiased=False)
            delta = batch_mean - self.mean
            total = self.count + b
            self.mean  = self.mean + delta * b / total
            m_a = self.var   * self.count
            m_b = batch_var  * b
            self.var   = (m_a + m_b + delta ** 2 * self.count * b / total) / total
            self.count = total
        std = torch.sqrt(self.var + 1e-5)
        return (x - self.mean) / std


# ──────────────────────────────────────────────────────────────────────────────
# Visual encoder
# ──────────────────────────────────────────────────────────────────────────────

def _conv_out(h_w, k, s):
    return (h_w[0] - k) // s + 1, (h_w[1] - k) // s + 1


class DepthCNN(nn.Module):
    """4-layer CNN for single-channel depth images.

    Input : (N, 1, H, W)
    Output: (N, 128, h, w)  – spatial feature map used as token sequence
    """

    def __init__(self, img_h=IMG_H, img_w=IMG_W):
        super().__init__()
        h, w = img_h, img_w
        h, w = _conv_out((h, w), 6, 2);  n1 = [16, h, w]
        h, w = _conv_out((h, w), 4, 2);  n2 = [32, h, w]
        h, w = _conv_out((h, w), 4, 2);  n3 = [64, h, w]
        h, w = _conv_out((h, w), 4, 2);  n4 = [128, h, w]
        self.num_tokens = h * w

        self.cnn = nn.Sequential(
            nn.Conv2d(1,  16,  kernel_size=6, stride=2),
            nn.ReLU(), nn.LayerNorm(n1),
            nn.Conv2d(16, 32,  kernel_size=4, stride=2),
            nn.ReLU(), nn.LayerNorm(n2),
            nn.Conv2d(32, 64,  kernel_size=4, stride=2),
            nn.ReLU(), nn.LayerNorm(n3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(), nn.LayerNorm(n4),
        )

    def forward(self, x):
        return self.cnn(x)          # (N, 128, h, w)


class DepthTransformerEncoder(nn.Module):
    """CNN + lightweight Transformer → scalar feature vector.

    Output: (N, CNN_OUT)
    """

    N_EMBD   = 128
    N_HEAD   = 4
    N_LAYERS = 2

    def __init__(self, img_h=IMG_H, img_w=IMG_W):
        super().__init__()
        self.cnn = DepthCNN(img_h, img_w)
        T = self.cnn.num_tokens
        d = self.N_EMBD

        self.pos_embed = nn.Embedding(T, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.N_HEAD,
            dim_feedforward=4 * d,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)

        self.out = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, CNN_OUT),
        )

    def forward(self, x, finetune=True):
        N = x.shape[0]
        if finetune:
            feat = self.cnn(x)
        else:
            with torch.no_grad():
                feat = self.cnn(x)

        # (N, 128, h*w) → (N, T, 128)
        feat = feat.flatten(2).transpose(1, 2)
        T = feat.shape[1]
        pos = self.pos_embed(torch.arange(T, device=x.device)).unsqueeze(0)
        feat = feat + pos

        cls = self.cls_token.expand(N, -1, -1)
        feat = torch.cat([cls, feat], dim=1)        # (N, T+1, 128)

        feat = self.transformer(feat)
        return self.out(feat[:, 0, :])              # (N, CNN_OUT)


# ──────────────────────────────────────────────────────────────────────────────
# Student model
# ──────────────────────────────────────────────────────────────────────────────

class DepthStudentModel(nn.Module):
    """Pure-PyTorch depth student for DAgger distillation.

    Parameters
    ----------
    num_obs     : proprioceptive observation dim
    num_actions : action dim
    num_seqs    : number of parallel envs (for RNN state init)
    mlp_units   : hidden units of the MLP
    rnn_units   : LSTM hidden size
    img_h, img_w: depth image resolution
    """

    is_aux = False          # checked by distillation.py

    def __init__(
        self,
        num_obs:     int,
        num_actions: int,
        num_seqs:    int   = 1,
        mlp_units:   tuple = (512, 512, 256),
        rnn_units:   int   = 512,
        img_h:       int   = IMG_H,
        img_w:       int   = IMG_W,
    ):
        super().__init__()
        self.num_seqs    = num_seqs
        self.rnn_units   = rnn_units
        self.num_actions = num_actions

        # Depth visual encoder
        self.depth_encoder   = DepthTransformerEncoder(img_h, img_w)
        self.running_mean_std = RunningMeanStd((1, img_h, img_w))

        # LSTM (before MLP)
        lstm_in = num_obs + CNN_OUT
        self.rnn      = nn.LSTM(lstm_in, rnn_units, batch_first=False)
        self.rnn_norm = nn.LayerNorm(rnn_units)

        # MLP
        layers = []
        in_size = rnn_units
        for u in mlp_units:
            layers += [nn.Linear(in_size, u), nn.ELU()]
            in_size = u
        self.mlp = nn.Sequential(*layers)

        # Output heads
        self.mu_head    = nn.Linear(in_size, num_actions)
        self.log_sigma  = nn.Parameter(torch.zeros(num_actions))
        self.value_head = nn.Linear(in_size, 1)

    # ------------------------------------------------------------------
    # Interface required by distillation.py
    # ------------------------------------------------------------------

    @property
    def a2c_network(self):
        """distillation.py checks `student_model.a2c_network.is_aux`."""
        return self

    def is_rnn(self):
        return True

    def get_default_rnn_state(self):
        n, u = 1, self.rnn_units
        return (
            torch.zeros(n, self.num_seqs, u),
            torch.zeros(n, self.num_seqs, u),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_dict: dict) -> dict:
        """
        batch_dict keys
        ---------------
        obs            : (N, num_obs)
        img            : (N, 1, H, W)  depth image  [optional]
        rnn_states     : list/tuple of LSTM h, c
        dones          : (N,) or (seq, N)  [optional]
        seq_length     : int  [default 1]
        finetune_backbone : bool  [default True]

        Returns
        -------
        dict with "mus", "sigmas", "values", "rnn_states"
        """
        obs       = batch_dict["obs"]
        states    = batch_dict.get("rnn_states", None)
        seq_len   = batch_dict.get("seq_length", 1)
        dones     = batch_dict.get("dones", None)
        finetune  = batch_dict.get("finetune_backbone", True)

        # ── depth encoding ──────────────────────────────────────────────
        if "img" in batch_dict:
            depth = batch_dict["img"]                       # (N, 1, H, W)
            with torch.no_grad():
                depth_n = self.running_mean_std(depth)
            depth_feat = self.depth_encoder(depth_n, finetune=finetune)
            obs = torch.cat([obs, depth_feat], dim=-1)      # (N, obs+CNN_OUT)

        # ── LSTM ────────────────────────────────────────────────────────
        N = obs.shape[0]
        num_seqs = N // seq_len

        # Unpack / default LSTM states
        if states is None or len(states) == 0:
            h = torch.zeros(1, N, self.rnn_units, device=obs.device)
            c = torch.zeros(1, N, self.rnn_units, device=obs.device)
        else:
            h, c = states[0], states[1]

        # Reshape to (seq_len, num_seqs, feat)
        x = obs.reshape(num_seqs, seq_len, -1).transpose(0, 1)

        # Apply done-masking: reset hidden state on episode boundaries
        if dones is not None:
            d = dones.reshape(num_seqs, seq_len).transpose(0, 1)   # (seq_len, num_seqs)
            h_t = h[:, :num_seqs, :]                                # (1, num_seqs, u)
            c_t = c[:, :num_seqs, :]
            outs = []
            for t in range(seq_len):
                mask = (1.0 - d[t].unsqueeze(0).unsqueeze(-1).float())
                h_t = h_t * mask
                c_t = c_t * mask
                out_t, (h_t, c_t) = self.rnn(x[t:t+1], (h_t, c_t))
                outs.append(out_t)
            rnn_out = torch.cat(outs, dim=0)    # (seq_len, num_seqs, u)
            new_h, new_c = h_t, c_t
        else:
            h_in = h[:, :num_seqs, :].contiguous()
            c_in = c[:, :num_seqs, :].contiguous()
            rnn_out, (new_h, new_c) = self.rnn(x, (h_in, c_in))

        rnn_out = rnn_out.transpose(0, 1).reshape(N, -1)   # (N, rnn_units)
        rnn_out = self.rnn_norm(rnn_out)

        # ── MLP + heads ─────────────────────────────────────────────────
        out    = self.mlp(rnn_out)
        mu     = self.mu_head(out)
        sigma  = self.log_sigma.exp().expand_as(mu)
        value  = self.value_head(out)

        # Expand states back to (1, num_seqs_original, u) shape
        h_out = new_h                   # (1, num_seqs, u)
        c_out = new_c

        return {
            "mus":        mu,
            "sigmas":     sigma,
            "values":     value,
            "rnn_states": (h_out, c_out),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Builder  (registered with model_builder.register_network in run_distillation.py)
# ──────────────────────────────────────────────────────────────────────────────

class DepthTransformerBuilder:
    """Thin builder wrapper so this model can be registered with RL-Games
    model_builder without importing rl_games inside this file.

    Usage in run_distillation.py
    ----------------------------
    from dextrah_lab.distillation.a2c_depth_transformer import DepthTransformerBuilder
    model_builder.register_network("a2c_depth_transformer", DepthTransformerBuilder)
    """

    def __init__(self, **kwargs):
        self.params = {}

    def load(self, params: dict):
        self.params = params

    def build(self, name, **kwargs):
        num_obs     = kwargs["input_shape"][0]
        num_actions = kwargs["actions_num"]
        num_seqs    = kwargs.get("num_seqs", 1)

        net_cfg  = self.params
        mlp_cfg  = net_cfg.get("mlp", {})
        rnn_cfg  = net_cfg.get("rnn", {})

        mlp_units = tuple(mlp_cfg.get("units", [512, 512, 256]))
        rnn_units = rnn_cfg.get("units", 512)
        img_h     = net_cfg.get("img_height", IMG_H)
        img_w     = net_cfg.get("img_width",  IMG_W)

        return DepthStudentModel(
            num_obs=num_obs,
            num_actions=num_actions,
            num_seqs=num_seqs,
            mlp_units=mlp_units,
            rnn_units=rnn_units,
            img_h=img_h,
            img_w=img_w,
        )

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
