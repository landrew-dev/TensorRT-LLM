# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import os
import argparse
import numpy as np
import tensorrt as trt

from .config import FastConformerConfig

LOGGER = trt.Logger(trt.Logger.WARNING)

def _add_subsampling(network: trt.INetworkDefinition,
                     feats: trt.ITensor,
                     W: dict[str, np.ndarray],
                     cfg: FastConformerConfig) -> trt.ITensor:
    """
    TODO(you): Implement your subsampling (e.g., 2D conv then reshape).
    For bring-up, you can pass-through or add a single 1x1 linear to match d_model.
    """
    # Minimal: a fully-connected projection feats[B,T,F] → [B,T,d_model]
    # TRT can't MatMul 3D directly; use shuffle to merge B and T then fullyConnected.
    BTF = network.add_shuffle(feats)
    BTF.reshape_dims = (-1, cfg.n_mels)
    fc_w = W.get("frontend.proj.weight")
    fc_b = W.get("frontend.proj.bias")
    if fc_w is None:
        # fallback: random frozen weights to enable engine build (replace later)
        fc_w = np.random.randn(cfg.d_model, cfg.n_mels).astype(np.float16)
        fc_b = np.zeros((cfg.d_model,), dtype=np.float16)
    fc = network.add_fully_connected(BTF.get_output(0), cfg.d_model, fc_w, fc_b)

    # Back to [B, T, d_model]
    BT_D = network.add_shuffle(fc.get_output(0))
    BT_D.reshape_dims = (-1, -1, cfg.d_model)
    return BT_D.get_output(0)

def _fastconformer_block(network: trt.INetworkDefinition,
                         x: trt.ITensor,
                         conv_state_in: trt.ITensor,
                         layer_id: int,
                         W: dict[str, np.ndarray],
                         cfg: FastConformerConfig) -> tuple[trt.ITensor, trt.ITensor]:
    """
    TODO(you): Port your block:
      x --LN→ MHA → +res
         --LN→ GLU+DepthwiseConv+SiLU+PW → +res
         --LN→ FFN → +res
    Maintain conv_state (carry K-1 samples/channel).
    Return (y, conv_state_out).

    For now, we forward x and identity state for bring-up.
    """
    # Identity “conv state” passthrough to make the streaming contract work now.
    conv_state_out = network.add_identity(conv_state_in).get_output(0)
    y = network.add_identity(x).get_output(0)
    return y, conv_state_out

def _ctc_head(network: trt.INetworkDefinition,
              x: trt.ITensor,
              W: dict[str, np.ndarray],
              cfg: FastConformerConfig) -> trt.ITensor:
    """
    Linear projection → logits [B, T, vocab]
    """
    BTD = network.add_shuffle(x); BTD.reshape_dims = (-1, cfg.d_model)
    w = W.get("ctc.weight")
    b = W.get("ctc.bias")
    if w is None:
        w = np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float16)
        b = np.zeros((cfg.vocab_size,), dtype=np.float16)
    fc = network.add_fully_connected(BTD.get_output(0), cfg.vocab_size, w, b)
    BTV = network.add_shuffle(fc.get_output(0)); BTV.reshape_dims = (-1, -1, cfg.vocab_size)
    return BTV.get_output(0)

def build_engine(cfg: FastConformerConfig, weights_npz: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with trt.Builder(LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as net, \
         builder.create_builder_config() as conf:

        if cfg.dtype == "fp16":
            conf.set_flag(trt.BuilderFlag.FP16)

        # Inputs
        feats = net.add_input("features", trt.float16, cfg.feats_shape())
        # Per-layer conv carry state inputs
        conv_state_in = []
        for l in range(cfg.num_layers):
            t = net.add_input(f"conv_state_{l}", trt.float16, cfg.conv_state_shape())
            conv_state_in.append(t)

        # Weights
        W = {k: v.astype(np.float16) for k, v in np.load(weights_npz).items()}

        # Graph
        x = _add_subsampling(net, feats, W, cfg)
        conv_state_out = []
        for l in range(cfg.num_layers):
            x, s_out = _fastconformer_block(net, x, conv_state_in[l], l, W, cfg)
            conv_state_out.append(s_out)
        logits = _ctc_head(net, x, W, cfg)

        # Outputs
        logits.name = "logits"
        net.mark_output(logits)
        for l, t in enumerate(conv_state_out):
            t.name = f"conv_state_{l}_out"
            net.mark_output(t)

        # One tight profile for bring-up
        prof = builder.create_optimization_profile()
        fmin = fopt = fmax = cfg.feats_shape()
        prof.set_shape("features", fmin, fopt, fmax)
        for l in range(cfg.num_layers):
            s = cfg.conv_state_shape()
            prof.set_shape(f"conv_state_{l}", s, s, s)
        conf.add_optimization_profile(prof)

        engine = builder.build_serialized_network(net, conf)
        out_path = os.path.join(out_dir, "encoder.plan")
        with open(out_path, "wb") as f:
            f.write(engine)
        print(f"[build] wrote {out_path}")
        return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = FastConformerConfig.from_file(args.config)
    build_engine(cfg, args.weights, args.out)

if __name__ == "__main__":
    main()
