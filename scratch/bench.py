from time import perf_counter
from tensorrt_llm import LLM, SamplingParams

def perf_to_dict(m, gen_len):
    tm = m.timing_metrics
    to_sec = lambda td: None if td is None else td.total_seconds()
    to_sec_diff = (
        lambda a, b: None if (a is None or b is None) else (a - b).total_seconds()
    )
    prefill_time_s = to_sec_diff(tm.first_token_time, tm.first_scheduled_time)
    decode_time_s = to_sec_diff(tm.last_token_time, tm.first_token_time)
    itl_den = max((gen_len or 0) - 1, 1)
    inter_token_latency_s = None if decode_time_s is None else decode_time_s / itl_den
    return {
        "prefill_time_ms": prefill_time_s * 1000,
        "decode_time_ms": decode_time_s * 1000,
        "inter_token_latency_ms": inter_token_latency_s * 1000,
        "timing": {
            "arrival_time_s": to_sec(tm.arrival_time),
            "first_scheduled_time_s": to_sec(tm.first_scheduled_time),
            "first_token_time_s": to_sec(tm.first_token_time),
            "last_token_time_s": to_sec(tm.last_token_time),
        },
        "first_iter": m.first_iter,
        "last_iter": m.last_iter,
        "iter": m.iter,
    }

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompts = ["Hello, my name is", "The capital of France is", "The future of AI is"]
    sp = SamplingParams(temperature=0.8, top_p=0.95, return_perf_metrics=True, max_tokens=64)

    for out in llm.generate(prompts, sp):
        m = out.outputs[0].request_perf_metrics
        gen_len = out.outputs[0].length
        print("Prompt len:", len(out.prompt_token_ids), "Gen len:", gen_len)
        print("Perf metrics:", perf_to_dict(m, gen_len))

if __name__ == "__main__":
    main()
