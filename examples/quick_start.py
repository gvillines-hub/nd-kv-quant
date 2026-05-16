"""
Quick start example: N+D quantization on synthetic data.

This example does not require a GPU. It generates synthetic KV-cache-like data,
applies N+D quantization and KIVI quantization, and compares attention output
quality. Use this to verify the package works before running real benchmarks.
"""

import os
import sys

# Allow running directly without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nd_kv_quant import (
    quant_kivi,
    quant_nd_kv,
    compute_output_cosine,
)


def main():
    # Generate synthetic KV cache data with outliers (typical of real attention)
    rng = np.random.default_rng(42)
    seq_len, head_dim = 128, 128
    K = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    
    # Inject outliers in specific dimensions (typical KV cache outlier pattern)
    K[:, 5] *= 10.0
    K[:, 17] *= 8.0
    V[3, :] *= 6.0
    
    fp16_bytes = seq_len * head_dim * 2 * 2  # K + V at fp16
    
    print("Comparison on synthetic outlier-heavy data:")
    print(f"  Sequence length: {seq_len}, head dim: {head_dim}")
    print()
    print(f"{'Method':<22} {'Compression':>12} {'Output Cosine':>15}")
    print("-" * 52)
    
    # FP16 baseline
    cos = compute_output_cosine(K, V, K, V)
    print(f"{'FP16 (no compress)':<22} {'1.0x':>12} {cos:>15.6f}")
    
    # KIVI 4-bit
    K_q, V_q, nbytes = quant_kivi(K, V, bits=4)
    cos = compute_output_cosine(K, V, K_q, V_q)
    ratio = fp16_bytes / nbytes
    print(f"{'KIVI 4-bit':<22} {f'{ratio:.1f}x':>12} {cos:>15.6f}")
    
    # N+D quality mode
    K_q, V_q, nbytes = quant_nd_kv(K, V, norm_bits=8, dir_bits=4)
    cos = compute_output_cosine(K, V, K_q, V_q)
    ratio = fp16_bytes / nbytes
    print(f"{'N8+D4 (quality)':<22} {f'{ratio:.1f}x':>12} {cos:>15.6f}")
    
    # N+D Pareto mode
    K_q, V_q, nbytes = quant_nd_kv(K, V, norm_bits=8, dir_bits=3)
    cos = compute_output_cosine(K, V, K_q, V_q)
    ratio = fp16_bytes / nbytes
    print(f"{'N8+D3 (Pareto)':<22} {f'{ratio:.1f}x':>12} {cos:>15.6f}")
    
    print()
    print("To run the real benchmark on a HuggingFace model:")
    print("  pip install transformers accelerate bitsandbytes")
    print("  python benchmarks/run_benchmarks.py")


if __name__ == "__main__":
    main()
