"""
Real-LLM benchmark: TurboQuant KV cache vs fp16 baseline.

Runs a short prompt-completion task on a small open model and reports:
    - actual packed KV cache memory
    - per-token attention output cosine similarity
    - generation perplexity / acceptance vs baseline

Default model: HuggingFaceTB/SmolLM2-360M-Instruct (small enough for
free Kaggle T4, big enough to be representative).

Usage:
    python -m benchmarks.bench_kv_real --model SmolLM2-360M --bits 4
    python -m benchmarks.bench_kv_real --bits 2 --bits_outlier 3 --n_outlier 32
"""

from __future__ import annotations

import argparse
import math
import os
import time

# Quieten the noisy weight-loading tqdm bar from transformers / accelerate.
# (When stdout is captured by a subprocess pipe, tqdm prints one line per
# update instead of redrawing in place, which floods the log.)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import torch

from tiny_turboquant import HybridTurboQuantKVCache, TurboQuantKVCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument(
        "--cache_type",
        choices=["compressed", "hybrid"],
        default="compressed",
        help="compressed = all packed; hybrid = recent fp16 + older packed",
    )
    p.add_argument("--mode", choices=["quick", "quality", "memory"], default="quality")

    # Backward-compatible v0.1 arguments.
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--bits_outlier", type=int, default=None,
                   help="v0.1-compatible outlier bit width for both K/V.")
    p.add_argument("--n_outlier", type=int, default=32)

    # v0.2 quality-aware hybrid arguments.
    p.add_argument("--recent_window", type=int, default=128,
                   help="Number of most recent tokens kept dense in hybrid mode.")
    p.add_argument("--key_bits", type=int, default=None)
    p.add_argument("--value_bits", type=int, default=None)
    p.add_argument("--key_outlier_bits", type=int, default=None)
    p.add_argument("--value_outlier_bits", type=int, default=None)
    p.add_argument("--n_key_outliers", type=int, default=64)
    p.add_argument("--n_value_outliers", type=int, default=32)
    p.add_argument("--global_calibration", action="store_true",
                   help="Use one global quantizer instead of per-layer calibration in hybrid mode.")

    p.add_argument("--prompt", default=(
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Spain is Madrid. The capital of Italy is"))
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = {"quick": 8, "quality": 16, "memory": 1}[args.mode]

    return args

@torch.no_grad()
def measure(model, tokenizer, prompt, cache, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        past_key_values=cache,
    )
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed, out


@torch.no_grad()
def measure_logit_kl(model, tokenizer, prompt, cache_factory, max_new_tokens=32):
    """Teacher-forced KL between fp16 baseline logits and TurboQuant logits.

    Both paths see identical input at every step. This isolates pure
    quantization error. Free-running generations are still produced for
    the "identical generation?" check, but they are NOT used for KL —
    otherwise the metric explodes the moment paths diverge by one token.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 1) Free-running baseline + TQ generations (for identical-generation check only)
    out_ref = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True,
    )
    cache = cache_factory()
    out_tq = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True,
        past_key_values=cache,
    )

    text_ref = tokenizer.decode(out_ref.sequences[0], skip_special_tokens=True)
    text_tq  = tokenizer.decode(out_tq.sequences[0],  skip_special_tokens=True)

    # First-divergence index: how many decode steps stay greedy-identical?
    # This is the production-relevant signal — small KL means the answer
    # token is identical even when later filler tokens drift.
    prompt_len_for_div = inputs["input_ids"].shape[1]
    ref_new = out_ref.sequences[0, prompt_len_for_div:].tolist()
    tq_new  = out_tq .sequences[0, prompt_len_for_div:].tolist()
    n_new   = min(len(ref_new), len(tq_new))
    first_diverge = next(
        (i for i in range(n_new) if ref_new[i] != tq_new[i]),
        n_new,
    )

    # 2) Teacher-forced KL: feed the SAME baseline-generated sequence into
    #    both fp16-cache and TQ-cache, compare per-position logits.
    seq = out_ref.sequences  # (1, prompt_len + new_tokens)
    prompt_len = inputs["input_ids"].shape[1]

    # fp16 reference: one forward pass, no cache compression.
    ref_logits = model(seq).logits  # (1, T, V)

    # TQ path: prefill + decode through TurboQuantKVCache so attention
    # is computed against quantised K/V at every step.
    tq_cache = cache_factory()
    # prefill on the prompt
    tq_logits_chunks = []
    out = model(seq[:, :prompt_len], past_key_values=tq_cache, use_cache=True)
    tq_logits_chunks.append(out.logits)
    # decode one token at a time, feeding the *baseline* tokens
    for t in range(prompt_len, seq.shape[1]):
        out = model(seq[:, t:t + 1], past_key_values=tq_cache, use_cache=True)
        tq_logits_chunks.append(out.logits)
    tq_logits = torch.cat(tq_logits_chunks, dim=1)  # (1, T, V)

    # Compare logits at positions [prompt_len-1 .. T-2] — the ones that
    # predict the new tokens. Both saw identical inputs.
    p = torch.softmax(ref_logits[:, prompt_len - 1:-1].float(), dim=-1)
    q = torch.softmax(tq_logits [:, prompt_len - 1:-1].float(), dim=-1).clamp_min_(1e-12)
    kls = (p * (p.clamp_min_(1e-12).log() - q.log())).sum(-1).squeeze(0).tolist()

    return text_ref, text_tq, kls, cache, first_diverge, n_new


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    import huggingface_hub.utils as hf_hub_utils

    # Hard-disable every progress bar transformers / huggingface_hub know
    # about. Env vars alone are unreliable when this script is launched as
    # a subprocess from kaggle_run.py (pipes confuse tqdm).
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    hf_hub_utils.disable_progress_bars()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    print(f"Loading {args.model}  ({args.dtype} on {args.device})")
    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(args.device).eval()

    def factory():
        if args.cache_type == "hybrid":
            return HybridTurboQuantKVCache(
                bits=args.bits,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                recent_window=args.recent_window,
                key_outlier_bits=(
                    args.key_outlier_bits
                    if args.key_outlier_bits is not None
                    else args.bits_outlier
                ),
                value_outlier_bits=(
                    args.value_outlier_bits
                    if args.value_outlier_bits is not None
                    else args.bits_outlier
                ),
                n_key_outliers=args.n_key_outliers,
                n_value_outliers=args.n_value_outliers,
                per_layer_calibration=not args.global_calibration,
            )

        return TurboQuantKVCache(
            bits=args.bits,
            bits_outlier=args.bits_outlier,
            n_outlier=args.n_outlier,
        )

    text_ref, text_tq, kls, cache, first_diverge, n_new = measure_logit_kl(
        model, tok, args.prompt, factory, max_new_tokens=args.max_new_tokens,
    )

    print("\n--- BASELINE (fp16, default DynamicCache) ---")
    print(text_ref)
    print("\n--- TURBOQUANT ---")
    print(text_tq)

    n_tokens = cache.get_seq_length(0)
    fp16_bytes = cache.fp16_baseline_bytes()
    actual_tq_bytes = cache.actual_memory_bytes()
    theoretical_tq_bytes = cache.theoretical_memory_bytes()

    print("\n--- METRICS ---")
    print(f"cache type             : {args.cache_type}")
    print(f"benchmark mode         : {args.mode}")
    print(f"prompt + decode tokens : {n_tokens}")
    if hasattr(cache, "compressed_seq_length"):
        print(f"compressed tokens      : {cache.compressed_seq_length(0)}")
        print(f"recent dense tokens    : {cache.recent_seq_length(0)}")
    print(f"fp16 KV cache actual   : {fp16_bytes / 1e6:.2f} MB")
    print(f"TurboQuant actual      : {actual_tq_bytes / 1e6:.2f} MB  "
          f"(× compression = {fp16_bytes / max(actual_tq_bytes, 1):.2f})")
    print(f"TurboQuant theoretical : {theoretical_tq_bytes / 1e6:.2f} MB")
    print(f"mean per-token logit KL: {sum(kls) / len(kls):.5f}  (teacher-forced; lower is better)")
    print(f"max  per-token logit KL: {max(kls):.5f}")
    print(f"first divergence at    : {first_diverge}/{n_new} decode tokens "
          f"(higher is better; {n_new}/{n_new} = exact match)")
    print(f"identical generation?  : {text_ref == text_tq}")
    print("note                  : measures memory-quality behavior, not faster inference")


if __name__ == "__main__":
    main()
