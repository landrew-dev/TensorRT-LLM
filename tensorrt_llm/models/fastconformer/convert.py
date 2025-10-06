# SPDX-License-Identifier: Apache-2.0
"""
Training ckpt → NPZ for TRT builder.
This keeps the mapping simple: you control the canonical NPZ keys
that your builder will load (e.g., "enc.layers.0.attn.qkv.weight", etc.)
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from .config import FastConformerConfig

def _move(t):
    return t.detach().cpu().float().numpy()

def convert_training_to_trtllm(ckpt_path: str, out_npz: str, cfg: FastConformerConfig):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    out: dict[str, np.ndarray] = {}

    for k in ["subsample.conv1.weight", "subsample.conv1.bias",
              "subsample.conv2.weight", "subsample.conv2.bias"]:
        if k in sd:
            out[f"{k}"] = _move(sd[k])

    for l in range(cfg.num_layers):
        prefix = f"encoder.layers.{l}."
        for leaf in ["attn.q_proj.weight", "attn.q_proj.bias",
                     "attn.k_proj.weight", "attn.k_proj.bias",
                     "attn.v_proj.weight", "attn.v_proj.bias",
                     "attn.out_proj.weight", "attn.out_proj.bias"]:
            k = prefix + leaf
            if k in sd: out[k] = _move(sd[k])

        for leaf in ["conv.pw1.weight", "conv.pw1.bias",
                     "conv.dw.weight",       # depthwise 1D
                     "conv.dw.bias",
                     "conv.pw2.weight", "conv.pw2.bias"]:
            k = prefix + leaf
            if k in sd: out[k] = _move(sd[k])

        for leaf in ["ffn.fc1.weight", "ffn.fc1.bias",
                     "ffn.fc2.weight", "ffn.fc2.bias"]:
            k = prefix + leaf
            if k in sd: out[k] = _move(sd[k])

        for leaf in ["ln_mha.weight", "ln_mha.bias",
                     "ln_conv.weight", "ln_conv.bias",
                     "ln_ffn.weight", "ln_ffn.bias"]:
            k = prefix + leaf
            if k in sd: out[k] = _move(sd[k])

    for k in ["ctc.weight", "ctc.bias"]:
        if k in sd: out[k] = _move(sd[k])

    np.savez(out_npz, **out)
    print(f"[convert] wrote {out_npz} with {len(out)} tensors.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = FastConformerConfig.from_file(args.config)
    convert_training_to_trtllm(args.ckpt, args.out, cfg)

if __name__ == "__main__":
    main()
