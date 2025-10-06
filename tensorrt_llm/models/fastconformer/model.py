# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...module import Module, ModuleList
from ...layers import LayerNorm, ColumnLinear
from ...mapping import Mapping
from .config import FastConformerConfig


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # [B, T, D] -> [B, H, T, d_k]
    B, T, D = x.shape
    d_k = D // num_heads
    return x.view(B, T, num_heads, d_k).transpose(1, 2).contiguous()

def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # [B, H, T, d_k] -> [B, T, D]
    B, H, T, d_k = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * d_k)


class ConvRingBuffer:
    """Ring buffer for 1D conv inputs (time-major). Stores (k-1) most recent inputs."""
    def __init__(self, receptive: int, channels: int):
        self.receptive = int(receptive)
        self.channels = int(channels)
        self.buf: Optional[torch.Tensor] = None

    def reset(self, batch_size: int):
        self.buf = None

    def concat(self, x_new: torch.Tensor) -> torch.Tensor:
        if self.receptive <= 0:
            return x_new
        if self.buf is None:
            self.buf = torch.zeros(
                x_new.size(0), self.receptive, self.channels,
                device=x_new.device, dtype=x_new.dtype
            )
        return torch.cat([self.buf, x_new], dim=1)

    def update(self, x_input: torch.Tensor):
        if self.receptive <= 0:
            return
        take = min(self.receptive, x_input.size(1))
        self.buf = x_input[:, -take:, :].detach()


class AttnKVCache:
    """Keeps last W_l timesteps of K,V for windowed self-attention (per layer)."""
    def __init__(self, num_heads: int, head_dim: int, window_left: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.Wl = int(window_left)
        self.K: Optional[torch.Tensor] = None
        self.V: Optional[torch.Tensor] = None

    def reset(self, batch_size: int):
        self.K, self.V = None, None

    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        if self.K is None:
            self.K, self.V = K_new, V_new
        else:
            self.K = torch.cat([self.K, K_new], dim=2)
            self.V = torch.cat([self.V, V_new], dim=2)
        if self.K.size(2) > self.Wl:
            self.K = self.K[:, :, -self.Wl:, :].detach()
            self.V = self.V[:, :, -self.Wl:, :].detach()

    def get(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.K, self.V


class StreamingSubsample8x(Module):
    """
    Three stride-2 DW-separable conv stages with ring buffers.
      in:  [B, T, F]
      out: [B, floor(T/8), D]
    """
    def __init__(self, config: FastConformerConfig):
        super().__init__()
        k = int(config.conv_kernel)
        pad = (k - 1) // 2
        C = int(config.sub_mid_channels)
        D = int(config.hidden_size)

        self.in_proj = ColumnLinear(
            config.feat_dim, C, bias=True, dtype=config.dtype,
            tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
            tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1,
        )

        self.dw1 = nn.Conv1d(C, C, k, stride=2, padding=pad, groups=C)
        self.pw1 = nn.Conv1d(C, C, 1)
        self.dw2 = nn.Conv1d(C, C, k, stride=2, padding=pad, groups=C)
        self.pw2 = nn.Conv1d(C, C, 1)
        self.dw3 = nn.Conv1d(C, C, k, stride=2, padding=pad, groups=C)
        self.pw3 = nn.Conv1d(C, C, 1)

        self.out_proj = ColumnLinear(
            C, D, bias=True, dtype=config.dtype,
            tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
            tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1,
        )

        rec = (k - 1)
        self.rb1 = ConvRingBuffer(rec, C)
        self.rb2 = ConvRingBuffer(rec, C)
        self.rb3 = ConvRingBuffer(rec, C)

    def reset_stream(self, batch_size: int):
        self.rb1.reset(batch_size)
        self.rb2.reset(batch_size)
        self.rb3.reset(batch_size)

    @torch.inference_mode()
    def forward_stream(self, feats_chunk: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(feats_chunk)             # [B,T,C]

        x1_in = self.rb1.concat(x)                # [B,T1_in,C]
        y = x1_in.transpose(1, 2)                 # [B,C,T1_in]
        y = F.silu(self.pw1(self.dw1(y)))
        y1 = y.transpose(1, 2)                    # [B,T1_out,C]
        self.rb1.update(x1_in)

        x2_in = self.rb2.concat(y1)
        y = x2_in.transpose(1, 2)
        y = F.silu(self.pw2(self.dw2(y)))
        y2 = y.transpose(1, 2)
        self.rb2.update(x2_in)

        x3_in = self.rb3.concat(y2)
        y = x3_in.transpose(1, 2)
        y = F.silu(self.pw3(self.dw3(y)))
        y3 = y.transpose(1, 2)
        self.rb3.update(x3_in)

        out = self.out_proj(y3)                   # [B,T/8,D]
        return out


class FastConformerEncoderLayer(Module):
    def __init__(self, config: FastConformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        D = config.hidden_size
        H = config.num_attention_heads
        k = config.conv_kernel
        ff = config.ff_mult * D
        self.layer_idx = layer_idx

        self.ln_ff1 = LayerNorm(normalized_shape=D, eps=config.norm_epsilon, dtype=config.dtype)
        self.ff1_up = ColumnLinear(D, ff, bias=config.bias, dtype=config.dtype,
                                   tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
                                   tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1)
        self.ff1_down = ColumnLinear(ff, D, bias=config.bias, dtype=config.dtype,
                                     tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
                                     tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1)

        assert D % H == 0, "hidden_size must be divisible by num_attention_heads"
        self.H = H
        self.d_k = D // H
        self.Wl = int(config.attn_window_left)

        self.ln_attn = LayerNorm(normalized_shape=D, eps=config.norm_epsilon, dtype=config.dtype)
        self.q_proj = ColumnLinear(D, D, bias=config.bias, dtype=config.dtype,
                                   tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
                                   tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1)
        self.k_proj = ColumnLinear(D, D, bias=config.bias, dtype=config.dtype,
                                   tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
                                   tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1)
        self.v_proj = ColumnLinear(D, D, bias=config.bias, dtype=config.dtype,
                                   tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
                                   tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1)
        self.o_proj = ColumnLinear(D, D, bias=config.bias, dtype=config.dtype,
                                   tp_group=config.mapping.tp_group if isinstance(config.mapping, Mapping) else None,
                                   tp_size=config.mapping.tp_size if isinstance(config.mapping, Mapping) else 1)

        pad = (k - 1) // 2
        self.ln_conv = LayerNorm(normalized_shape=D, eps=config.norm_epsilon, dtype=config.dtype)
        self.pw1 = nn.Conv1d(D, 2 * D, 1)
        self.dw  = nn.Conv1d(2 * D, 2 * D, k, padding=pad, groups=2 * D)
        self.bn  = nn.BatchNorm1d(2 * D)
        self.pw2 = nn.Conv1d(2 * D, D, 1)

        self.conv_rb = ConvRingBuffer(receptive=(k - 1), channels=D)
        self.kv = AttnKVCache(self.H, self.d_k, self.Wl)

    def reset_stream(self, batch_size: int):
        self.kv.reset(batch_size)
        self.conv_rb.reset(batch_size)

    @torch.inference_mode()
    def forward_stream(self, x_new: torch.Tensor) -> torch.Tensor:
        # x_new: [B, T_new, D]
        B, T_new, D = x_new.shape

        # FF/2
        y = self.ln_ff1(x_new)
        y = self.ff1_down(F.silu(self.ff1_up(y)))
        x = x_new + 0.5 * y

        # attention (windowed, non-causal)
        y = self.ln_attn(x)
        Q = _split_heads(self.q_proj(y), self.H)          # [B,H,T_new,d_k]
        K_new = _split_heads(self.k_proj(y), self.H)
        V_new = _split_heads(self.v_proj(y), self.H)

        K_ctx, V_ctx = self.kv.get()
        if K_ctx is None:
            K_all, V_all = K_new, V_new
        else:
            K_all = torch.cat([K_ctx, K_new], dim=2)
            V_all = torch.cat([V_ctx, V_new], dim=2)

        attn = F.scaled_dot_product_attention(Q, K_all, V_all, is_causal=False)
        attn = _merge_heads(attn)                          # [B,T_new,D]
        x = x + self.o_proj(attn)

        self.kv.append(K_new, V_new)

        y_in = self.conv_rb.concat(self.ln_conv(x))       # [B, T_ctx+T_new, D]
        y = y_in.transpose(1, 2)                          # [B,D,T]
        y = self.pw1(y)
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y).transpose(1, 2)                   # [B,T,D]
        y = y[:, -T_new:, :]
        x = x + y
        self.conv_rb.update(y_in)

        y = self.ff1_down(F.silu(self.ff1_up(x)))
        x = x + 0.5 * y

        return x


class FastConformerEncoder(Module):
    """Streaming Fast-Conformer encoder with 8x subsampling frontend."""
    config_class = FastConformerConfig

    def __init__(self, config: FastConformerConfig):
        super().__init__()
        self.config = config
        self.mapping = config.mapping if isinstance(config.mapping, Mapping) else Mapping()

        self.sub = StreamingSubsample8x(config)
        self.layers = ModuleList(
            [FastConformerEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.ln_out = LayerNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def reset_stream(self, batch_size: int):
        self.sub.reset_stream(batch_size)
        for b in self.layers:
            b.reset_stream(batch_size)

    @torch.inference_mode()
    def forward_chunk(self, feats_chunk: torch.Tensor) -> torch.Tensor:
        """
        feats_chunk: [B, T_in, F]  (e.g., mel frames @10ms hop)
        returns:     [B, floor(T_in/8), D]
        """
        x = self.sub.forward_stream(feats_chunk)   # [B, T/8, D]
        for b in self.layers:
            x = b.forward_stream(x)
        return self.ln_out(x)
