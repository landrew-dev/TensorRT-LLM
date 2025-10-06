# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

from tensorrt_llm.models.fastconformer.config import FastConformerConfig

@dataclass
class Perf:
    first_chunk_ms: float | None = None
    chunk_ms: List[float] = None
    total_audio_s: float = 0.0

    def rtf(self):
        proc = (sum(self.chunk_ms) if self.chunk_ms else 0.0) / 1000.0
        return proc / max(self.total_audio_s, 1e-9)

class GreedyCTC:
    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id
        self.prev_token = None
        self.out: List[int] = []

    def step(self, logits_b_t_v: np.ndarray):
        # logits: [1, T_out, V] (fp16)
        ids = logits_b_t_v.argmax(axis=-1).astype(np.int32)[0]  # [T_out]
        for tok in ids:
            if tok != self.blank_id and tok != self.prev_token:
                self.out.append(int(tok))
            self.prev_token = int(tok)

    def finalize(self) -> List[int]:
        return self.out

class FastConformerSession:
    def __init__(self, engine_path: str, cfg: FastConformerConfig):
        self.cfg = cfg
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.bind_idx = {self.engine.get_binding_name(i): i for i in range(self.engine.num_bindings)}

        self.state: Dict[str, np.ndarray] = {
            f"conv_state_{l}": np.zeros(cfg.conv_state_shape(), dtype=np.float16)
            for l in range(cfg.num_layers)
        }

    def _set_shapes(self, feats: np.ndarray):
        self.ctx.set_binding_shape(self.bind_idx["features"], feats.shape)
        for l in range(self.cfg.num_layers):
            name = f"conv_state_{l}"
            self.ctx.set_binding_shape(self.bind_idx[name], self.state[name].shape)

    def infer_chunk(self, feats_chunk_fp16: np.ndarray) -> Tuple[np.ndarray, float]:
        assert feats_chunk_fp16.shape == self.cfg.feats_shape(), \
            f"expected {self.cfg.feats_shape()}, got {feats_chunk_fp16.shape}"

        self._set_shapes(feats_chunk_fp16)

        d_bindings = [None] * self.engine.num_bindings

        d_feats = cuda.mem_alloc(feats_chunk_fp16.nbytes)
        cuda.memcpy_htod(d_feats, feats_chunk_fp16)
        d_bindings[self.bind_idx["features"]] = int(d_feats)

        d_state_in: Dict[str, cuda.DeviceAllocation] = {}
        for l in range(self.cfg.num_layers):
            name = f"conv_state_{l}"
            host = self.state[name]
            d = cuda.mem_alloc(host.nbytes)
            cuda.memcpy_htod(d, host)
            d_state_in[name] = d
            d_bindings[self.bind_idx[name]] = int(d)

        T_out = feats_chunk_fp16.shape[1] // self.cfg.subsampling_factor
        out_logits_shape = (1, max(T_out, 1), self.cfg.vocab_size)
        d_logits = cuda.mem_alloc(np.prod(out_logits_shape) * np.dtype(np.float16).itemsize)
        d_bindings[self.bind_idx["logits"]] = int(d_logits)

        d_state_out: Dict[str, Tuple[cuda.DeviceAllocation, tuple]] = {}
        for l in range(self.cfg.num_layers):
            name = f"conv_state_{l}_out"
            shape = self.cfg.conv_state_shape()
            d = cuda.mem_alloc(np.prod(shape) * 2)
            d_state_out[name] = (d, shape)
            d_bindings[self.bind_idx[name]] = int(d)

        start_evt, end_evt = cuda.Event(), cuda.Event()
        start_evt.record()
        self.ctx.execute_v2(d_bindings)
        end_evt.record(); end_evt.synchronize()
        ms = start_evt.time_till(end_evt)

        logits = np.empty(out_logits_shape, dtype=np.float16)
        cuda.memcpy_dtoh(logits, d_logits)

        next_state: Dict[str, np.ndarray] = {}
        for name, (d, shape) in d_state_out.items():
            h = np.empty(shape, dtype=np.float16)
            cuda.memcpy_dtoh(h, d)
            next_state[name.replace("_out", "")] = h
        self.state = next_state

        return logits, ms
