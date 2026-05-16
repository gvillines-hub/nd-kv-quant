"""
Reproducible benchmark: N+D vs KIVI vs TurboQuant on a real model.

Loads a HuggingFace transformer model, extracts the KV cache from a long-context
prompt, then runs all compression methods and reports:

  - Compression ratio (vs FP16)
  - Average attention output cosine similarity
  - 5th-percentile cosine
  - Minimum cosine (the single worst-case layer/head)

The minimum cosine is the key diagnostic metric. Average performance is roughly
equivalent across quantization methods. The interesting question is how badly
the worst-case layer degrades.

Methods tested:
  - KIVI 4-bit / 2-bit baselines
  - N+D N8+D4 / D3 / D2 (per-column min-max on direction components)
  - N+D shared codebook variants (Lloyd-Max ablation: same as N+D but with a
    single empirical codebook for all direction values, testing whether
    per-column adaptation is doing real work)
  - TurboQuant uniform (random rotation + WHT)
  - TurboQuant outlier-aware mixed precision

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --model meta-llama/Llama-3.1-8B
    python benchmarks/run_benchmarks.py --no-4bit
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np

from nd_kv_quant import (
    quant_kivi,
    quant_nd_kv,
    compute_output_cosine,
)
from nd_kv_quant.extraction import extract_kv_cache
from nd_kv_quant.turboquant import quant_turboquant_kv
from nd_kv_quant.turboquant_outlier import quant_turboquant_outlier_aware_kv
from nd_kv_quant.nd_shared import quant_nd_kv_shared


def classify_kivi_health(min_cos: float) -> str:
    if min_cos >= 0.95:
        return "HEALTHY"
    elif min_cos >= 0.85:
        return "DEGRADED"
    else:
        return "CATASTROPHIC"


def main():
    parser = argparse.ArgumentParser(description="N+D vs KIVI vs TurboQuant benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HuggingFace model ID")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--output", default="benchmark_results.json", help="Save results JSON")
    args = parser.parse_args()

    print("=" * 80)
    print("ND-KV-QUANT BENCHMARK")
    print("=" * 80)
    print(f"Model:    {args.model}")
    print(f"4-bit:    {not args.no_4bit}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Extracting KV cache from model...")
    t0 = time.time()
    all_K, all_V, n_layers, n_heads, seq_len = extract_kv_cache(
        model_name=args.model, use_4bit=not args.no_4bit
    )
    head_dim = all_K[0].shape[2]
    print(f"  {n_layers} layers x {n_heads} KV heads x {head_dim} dim x {seq_len} tokens")
    print(f"  Extraction time: {time.time() - t0:.1f}s")
    print()

    methods = [
        ("FP16 baseline",                lambda K, V: (K.copy(), V.copy(), seq_len * head_dim * 2 * 2)),
        ("KIVI 4-bit",                   lambda K, V: quant_kivi(K, V, 4)),
        ("KIVI 2-bit",                   lambda K, V: quant_kivi(K, V, 2)),
        ("N8+D4 (per-col, quality)",     lambda K, V: quant_nd_kv(K, V, 8, 4)),
        ("N8+D3 (per-col, Pareto)",      lambda K, V: quant_nd_kv(K, V, 8, 3)),
        ("N8+D2 (per-col, aggressive)",  lambda K, V: quant_nd_kv(K, V, 8, 2)),
        ("N8+D4 (shared codebook)",      lambda K, V: quant_nd_kv_shared(K, V, 8, 4)),
        ("N8+D3 (shared codebook)",      lambda K, V: quant_nd_kv_shared(K, V, 8, 3)),
        ("N8+D2 (shared codebook)",      lambda K, V: quant_nd_kv_shared(K, V, 8, 2)),
        ("TurboQuant 4-bit (rand)",      lambda K, V: quant_turboquant_kv(K, V, 4, rotation="random")),
        ("TurboQuant 4-bit (WHT)",       lambda K, V: quant_turboquant_kv(K, V, 4, rotation="wht")),
        ("TurboQuant 3-bit (rand)",      lambda K, V: quant_turboquant_kv(K, V, 3, rotation="random")),
        ("TurboQuant 3-bit (WHT)",       lambda K, V: quant_turboquant_kv(K, V, 3, rotation="wht")),
        ("TurboQuant 2-bit (rand)",      lambda K, V: quant_turboquant_kv(K, V, 2, rotation="random")),
        ("TurboQuant 2-bit (WHT)",       lambda K, V: quant_turboquant_kv(K, V, 2, rotation="wht")),
        ("TQ outlier 3.5b (10%@8+3)",    lambda K, V: quant_turboquant_outlier_aware_kv(K, V, 8, 3, 0.10)),
        ("TQ outlier 3.2b (20%@4+3)",    lambda K, V: quant_turboquant_outlier_aware_kv(K, V, 4, 3, 0.20)),
        ("TQ outlier 2.25b (25%@3+2)",   lambda K, V: quant_turboquant_outlier_aware_kv(K, V, 3, 2, 0.25)),
    ]

    results = {}

    print(f"{'Method':<32} {'Comp':>6} {'AvgCos':>10} {'P5':>8} {'Min':>8}")
    print("-" * 70)

    for name, fn in methods:
        total_orig = 0
        total_comp = 0
        cosines = []

        for l in range(n_layers):
            for h in range(n_heads):
                K = all_K[l][h]
                V = all_V[l][h]
                fp16_bytes = seq_len * head_dim * 2 * 2
                total_orig += fp16_bytes

                K_c, V_c, c_bytes = fn(K, V)
                total_comp += c_bytes
                cosines.append(compute_output_cosine(K, V, K_c, V_c))

        comp_ratio = total_orig / total_comp
        avg_cos = float(np.mean(cosines))
        p5_cos = float(np.percentile(cosines, 5))
        min_cos = float(np.min(cosines))

        flag = ""
        if name.startswith("N8+D") and "per-col" in name and "KIVI 4-bit" in results:
            kivi = results["KIVI 4-bit"]
            if avg_cos > kivi["avg_cos"] and min_cos > kivi["min_cos"] and comp_ratio >= 3.5:
                flag = "  <- beats KIVI"

        print(f"  {name:<30} {comp_ratio:>5.1f}x {avg_cos:>10.4f} {p5_cos:>8.4f} {min_cos:>8.4f}{flag}")

        results[name] = {
            "compression_ratio": comp_ratio,
            "avg_cos": avg_cos,
            "p5_cos": p5_cos,
            "min_cos": min_cos,
        }

    kivi_min = results["KIVI 4-bit"]["min_cos"]
    nd_min = results["N8+D4 (per-col, quality)"]["min_cos"]
    gap = nd_min - kivi_min
    health = classify_kivi_health(kivi_min)

    print()
    print("=" * 80)
    print(f"DIAGNOSTIC: KIVI 4-bit worst-case behavior on {args.model}")
    print("=" * 80)
    print(f"  KIVI 4-bit min cosine:  {kivi_min:.4f}")
    print(f"  N8+D4 min cosine:       {nd_min:.4f}")
    print(f"  Gap (N+D advantage):    {gap:+.4f}")
    print(f"  Verdict:                {health}")
    print()

    # Shared codebook ablation summary
    print("=" * 80)
    print("ABLATION: N+D per-column vs shared codebook")
    print("=" * 80)
    for b in [4, 3, 2]:
        per_col = results.get(f"N8+D{b} (per-col, " + ("quality" if b == 4 else "Pareto" if b == 3 else "aggressive") + ")", None)
        shared = results.get(f"N8+D{b} (shared codebook)", None)
        if per_col and shared:
            print(f"  b={b}: per-col min {per_col['min_cos']:.4f} vs shared min {shared['min_cos']:.4f} "
                  f"(gap: {per_col['min_cos'] - shared['min_cos']:+.4f})")

    output_data = {
        "model": args.model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "timestamp": datetime.now().isoformat(),
        "diagnostic": {
            "kivi_4bit_min_cos": kivi_min,
            "nd_n8d4_min_cos": nd_min,
            "gap": gap,
            "verdict": health,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
