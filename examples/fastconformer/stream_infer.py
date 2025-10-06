# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse
import numpy as np

from tensorrt_llm.models.fastconformer.config import FastConformerConfig
from tensorrt_llm.runtime.fastconformer_runtime import FastConformerSession, GreedyCTC, Perf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path/to/encoder.plan")
    ap.add_argument("--config", required=True, help="path/to/fastconformer.json")
    ap.add_argument("--nchunks", type=int, default=10, help="how many chunks to stream")
    args = ap.parse_args()

    cfg = FastConformerConfig.from_file(args.config)

    sess = FastConformerSession(args.engine, cfg)
    ctc = GreedyCTC(blank_id=cfg.blank_id)
    perf = Perf(chunk_ms=[], total_audio_s=(cfg.chunk_size * args.nchunks * 0.01))  # assume 10ms hop

    print(f"[info] Running {args.nchunks} random chunks of shape {cfg.feats_shape()} ...")

    first = True
    for i in range(args.nchunks):
        # generate random fake features [1, T_chunk, n_mels]
        feats = np.random.randn(*cfg.feats_shape()).astype(np.float16)
        logits, ms = sess.infer_chunk(feats)
        if first:
            perf.first_chunk_ms = ms
            first = False
        perf.chunk_ms.append(ms)
        ctc.step(logits)
        print(f"[chunk {i+1:02d}] {ms:.2f} ms, partial len={len(ctc.out)}", flush=True)

    ids = ctc.finalize()
    print("\n==== RESULTS ====")
    print(f"First chunk latency: {perf.first_chunk_ms:.2f} ms")
    print(f"Avg chunk latency: {np.mean(perf.chunk_ms):.2f} ms   (N={len(perf.chunk_ms)})")
    print(f"RTF: {perf.rtf():.3f}")
    print(f"IDs: {ids[:64]}{' ...' if len(ids)>64 else ''}")

if __name__ == "__main__":
    main()
