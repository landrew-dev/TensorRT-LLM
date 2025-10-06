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

from typing import Optional, Union

from ...logger import logger
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class FastConformerConfig(PretrainedConfig):
    """Configuration for a streaming Fast-Conformer encoder."""

    def __init__(
        self,
        *,
        architecture: str = "FastConformerEncoder",
        dtype: str = "float16",
        feat_dim: int = 80,
        subsample: tuple = (2, 2, 2),
        sub_mid_channels: int = 256,
        conv_kernel: int = 9,
        hidden_size: int = 512,
        num_hidden_layers: int = 17,
        num_attention_heads: int = 8,
        ff_mult: int = 4,
        attn_window_left: int = 70,
        norm_epsilon: float = 1e-5,
        bias: bool = True,
        mapping: Optional[Mapping] = None,
        quantization: Optional[QuantConfig] = None,
        **kwargs,
    ):
        self.architecture = architecture
        self.dtype = dtype

        self.feat_dim = feat_dim
        self.subsample = tuple(subsample)
        self.sub_mid_channels = sub_mid_channels
        self.conv_kernel = conv_kernel

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ff_mult = ff_mult
        self.attn_window_left = attn_window_left

        self.norm_epsilon = norm_epsilon
        self.bias = bias

        super().__init__(
            dtype=dtype,
            mapping=mapping,
            quantization=quantization,
            **kwargs,
        )

    def to_dict(self):
        out = super().to_dict()
        out.update({
            "architecture": self.architecture,
            "feat_dim": self.feat_dim,
            "subsample": self.subsample,
            "sub_mid_channels": self.sub_mid_channels,
            "conv_kernel": self.conv_kernel,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "ff_mult": self.ff_mult,
            "attn_window_left": self.attn_window_left,
            "norm_epsilon": self.norm_epsilon,
            "bias": self.bias,
        })
        return out

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        raise NotImplementedError

    @classmethod
    def from_nemo(cls,
                  nemo_ckpt_dir: str,
                  dtype: str = 'auto',
                  mapping: Optional[Mapping] = None,
                  quant_config: Optional[QuantConfig] = None,
                  **kwargs):
        raise NotImplementedError
