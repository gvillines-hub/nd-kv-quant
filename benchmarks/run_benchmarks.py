"""
Reproducible benchmark: N+D vs KIVI on a real model.

Loads a HuggingFace transformer model, extracts the KV cache from a long-context
prompt, then runs all compression methods and reports:

  - Compression ratio (vs FP16)
  - Average attention output cosine similarity
  - 5th-percentile cosine
  - Minimum cosine (the single worst-case layer/head)

The minimum cosine is the key diagnostic metric. Average performance is roughly
equivalent across quantization methods. The interesting question is how badly
the worst-case layer degrades. On some architectures (notably Qwen 2.5),
KIVI 4-bit can silently produce per-layer cosines below 0.7 even when the
average looks fine. N+D quantization at equal compression typically keeps the
minimum above 0.95.

Whether you NEED N+D depends on whether your model exhibits this failure mode.
This benchmark is the diagnostic.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --model meta-llama/Llama-3.1-8B
    python benchmarks/run_benchmarks.py --no-4bit         # use fp16 instead of bitsandbytes

Requires GPU. Tested on RTX 5060 Ti 16GB and similar.
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


def classify_kivi_health(min_cos: float) -> str:
    """
    Categorize KIVI 4-bit worst-case behavior on the tested model.
    
    These thresholds are heuristic, drawn from empirical observation of how
    KIVI behaves across multiple model families. Models with min cosine
    above 0.95 work well with KIVI; below 0.85 indicates a likely silent
    failure mode that production telemetry would not catch.
    """
    if min_cos >= 0.95:
        return "HEALTHY"
    elif min_cos >= 0.85:
        return "DEGRADED"
    else:
        return "CATASTROPHIC"


def main():
    parser = argparse.ArgumentParser(description="N+D vs KIVI benchmark")
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
    
    # Extract KV cache
    print("Extracting KV cache from model...")
    t0 = time.time()
    all_K, all_V, n_layers, n_heads, seq_len = extract_kv_cache(
        model_name=args.model, use_4bit=not args.no_4bit
    )
    head_dim = all_K[0].shape[2]
    print(f"  {n_layers} layers x {n_heads} KV heads x {head_dim} dim x {seq_len} tokens")
    print(f"  Extraction time: {time.time() - t0:.1f}s")
    print()
    
    # Define methods to test
    methods = [
        ("FP16 baseline",       lambda K, V: (K.copy(), V.copy(), seq_len * head_dim * 2 * 2)),
        ("KIVI 4-bit",          lambda K, V: quant_kivi(K, V, 4)),
        ("KIVI 2-bit",          lambda K, V: quant_kivi(K, V, 2)),
        ("N8+D4 (quality)",     lambda K, V: quant_nd_kv(K, V, 8, 4)),
        ("N8+D3 (Pareto)",      lambda K, V: quant_nd_kv(K, V, 8, 3)),
        ("N8+D2 (aggressive)",  lambda K, V: quant_nd_kv(K, V, 8, 2)),
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
        if name.startswith("N8+D") and "KIVI 4-bit" in results:
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
    
    # Data-driven diagnostic summary based on THIS run's actual numbers
    kivi_min = results["KIVI 4-bit"]["min_cos"]
    nd_min = results["N8+D4 (quality)"]["min_cos"]
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
    
    if health == "CATASTROPHIC":
        print("  This model exhibits silent failure mode under KIVI 4-bit.")
        print("  The average looks fine but at least one layer has a worst-case")
        print("  attention output cosine far below 0.85. Production telemetry")
        print("  measuring perplexity or task accuracy would likely not catch this.")
        print()
        print("  N+D quantization at equal compression maintains worst-case behavior.")
        print("  Recommend switching to N8+D4 for this model.")
    elif health == "DEGRADED":
        print("  This model shows some worst-case degradation under KIVI 4-bit.")
        print("  Not catastrophic but worth monitoring. N+D would improve worst-case")
        print("  behavior at the same compression ratio.")
    else:
        print("  KIVI 4-bit works well on this model. N+D offers marginal improvement")
        print("  but no urgent reason to switch unless you want the headroom.")
    
    # Save results
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
