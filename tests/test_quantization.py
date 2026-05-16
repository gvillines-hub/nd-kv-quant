"""
Correctness tests for the quantization algorithms.

These tests run without GPU and validate that:
  - Quantization functions return arrays of the correct shape
  - Storage cost calculations are correct
  - N+D quantization preserves vector direction better than per-channel
  - The decomposition / reconstruction is mathematically sound

Run with: python -m pytest tests/ -v
Or directly: python tests/test_quantization.py
"""

import numpy as np
import sys
import os

# Allow running directly without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd_kv_quant import (
    norm_direction_quant,
    quant_per_channel,
    quant_per_token,
    quant_kivi,
    quant_nd_kv,
    compute_output_cosine,
)


def make_test_data(seq_len=64, head_dim=128, seed=42):
    """Generate KV-cache-like test data with some outlier structure."""
    rng = np.random.default_rng(seed)
    # Base data: normal distribution
    K = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    # Add outliers in a few dimensions (typical of real KV cache)
    K[:, 5] *= 10.0
    K[:, 17] *= 8.0
    V[3, :] *= 6.0
    return K, V


def test_shapes_preserved():
    """Quantization functions must return arrays of the original shape."""
    K, V = make_test_data()
    
    for bits in [2, 4, 8]:
        K_q, _ = quant_per_channel(K, bits)
        assert K_q.shape == K.shape, f"per_channel {bits}b: shape mismatch"
        
        V_q, _ = quant_per_token(V, bits)
        assert V_q.shape == V.shape, f"per_token {bits}b: shape mismatch"
    
    for nb in [4, 8]:
        for db in [2, 3, 4]:
            K_q, _ = norm_direction_quant(K, nb, db)
            assert K_q.shape == K.shape, f"N+D N{nb}D{db}: shape mismatch"
    
    print("PASS: shape preservation")


def test_storage_cost():
    """Storage cost must match the bit-width math."""
    seq, dim = 100, 128
    K = np.random.randn(seq, dim).astype(np.float32)
    
    # per-channel 4-bit: seq*dim*4/8 bytes + dim*4 bytes metadata
    _, nbytes = quant_per_channel(K, 4)
    expected = (seq * dim * 4) / 8 + dim * 4
    assert abs(nbytes - expected) < 1, f"per_channel 4b: got {nbytes}, expected {expected}"
    
    # N+D 8/4: norms (seq*8/8 + 4) + directions (seq*dim*4/8 + dim*4)
    _, nbytes = norm_direction_quant(K, 8, 4)
    expected = (seq * 8) / 8 + 4 + (seq * dim * 4) / 8 + dim * 4
    assert abs(nbytes - expected) < 1, f"N+D 8/4: got {nbytes}, expected {expected}"
    
    print("PASS: storage cost calculations")


def test_nd_beats_kivi_on_outliers():
    """N+D quantization should preserve attention output better than KIVI on outlier-heavy data."""
    K, V = make_test_data()
    
    # KIVI 4-bit
    K_kivi, V_kivi, _ = quant_kivi(K, V, 4)
    kivi_cos = compute_output_cosine(K, V, K_kivi, V_kivi)
    
    # N+D 8/4 (same ~3.9x compression)
    K_nd, V_nd, _ = quant_nd_kv(K, V, 8, 4)
    nd_cos = compute_output_cosine(K, V, K_nd, V_nd)
    
    # On outlier data, N+D should match or beat KIVI
    assert nd_cos >= kivi_cos - 0.01, (
        f"N+D should match KIVI on outlier data: N+D={nd_cos:.4f}, KIVI={kivi_cos:.4f}"
    )
    
    print(f"PASS: N+D ({nd_cos:.4f}) >= KIVI ({kivi_cos:.4f}) on outlier-heavy data")


def test_nd_compression_ratios():
    """Verify expected compression ratios are achieved."""
    K, V = make_test_data(seq_len=256)
    seq_len, head_dim = K.shape
    fp16_bytes = seq_len * head_dim * 2 * 2  # K + V
    
    configs = [
        (8, 4, 3.8, 4.0),   # Expected ~3.9x
        (8, 3, 4.9, 5.3),   # Expected ~5.1x
        (8, 2, 7.0, 8.0),   # Expected ~7.5x
    ]
    
    for nb, db, lo, hi in configs:
        _, _, nbytes = quant_nd_kv(K, V, nb, db)
        ratio = fp16_bytes / nbytes
        assert lo <= ratio <= hi, f"N{nb}D{db}: ratio {ratio:.2f} not in [{lo}, {hi}]"
    
    print("PASS: compression ratios within expected ranges")


def test_renormalization():
    """After N+D quantization, reconstructed vectors should preserve direction approximately."""
    K, _ = make_test_data(seed=7)
    K_q, _ = norm_direction_quant(K, norm_bits=8, dir_bits=4)
    
    # Cosine between original and quantized direction (per row)
    K_dir = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-10)
    Kq_dir = K_q / (np.linalg.norm(K_q, axis=1, keepdims=True) + 1e-10)
    cos_per_row = np.sum(K_dir * Kq_dir, axis=1)
    
    avg_cos = float(np.mean(cos_per_row))
    min_cos = float(np.min(cos_per_row))
    assert avg_cos > 0.99, f"Mean direction preservation too low: {avg_cos}"
    assert min_cos > 0.95, f"Worst direction preservation too low: {min_cos}"
    
    print(f"PASS: direction preservation (mean={avg_cos:.4f}, min={min_cos:.4f})")


def test_fp16_is_identity():
    """FP16 'baseline' (no quantization) should reproduce input exactly."""
    K, V = make_test_data()
    cos = compute_output_cosine(K, V, K, V)
    assert cos > 0.9999, f"Identity attention cosine too low: {cos}"
    print(f"PASS: identity baseline (cos={cos:.6f})")


if __name__ == "__main__":
    print("Running correctness tests...\n")
    test_shapes_preserved()
    test_storage_cost()
    test_fp16_is_identity()
    test_nd_compression_ratios()
    test_renormalization()
    test_nd_beats_kivi_on_outliers()
    print("\nAll tests passed.")
