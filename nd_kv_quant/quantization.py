"""
Core quantization functions for KV cache compression.

Implements:
  - Norm + Direction (N+D) quantization: decomposes vectors into scalar magnitude
    and unit direction, quantizes each independently. Based on the magnitude-direction
    decomposition introduced by MiniCache (Liu et al., NeurIPS 2024) and the
    polar-coordinate quantization analysis from PolarQuant (Han et al., 2025).
    
    This implementation uses a simpler per-column scalar quantization of the unit
    direction vector rather than recursive polar transformation. The empirical
    finding is that quantization error in the direction component is approximately
    isotropic with respect to dot-product attention, which prevents the catastrophic
    worst-case layer degradation observed with KIVI-style per-channel quantization.

  - KIVI baselines: per-channel quantization for keys, per-token quantization for
    values. Implemented as comparison baseline. Based on Liu et al., 2024.
"""

import numpy as np
from typing import Tuple


def quant_per_channel(data: np.ndarray, bits: int) -> Tuple[np.ndarray, float]:
    """
    Per-channel (per-column) min/max quantization.
    
    Each column (feature dimension) is quantized independently using its own
    min and max values. This is the KIVI strategy for key vectors.
    
    Args:
        data: Array of shape (seq_len, dim) to quantize
        bits: Number of bits for quantization (typically 2, 4, or 8)
    
    Returns:
        quantized: Dequantized array with same shape as input
        nbytes: Storage cost in bytes (quantized values + per-column scale/zero metadata)
    """
    seq, dim = data.shape
    result = np.zeros_like(data)
    for d in range(dim):
        col = data[:, d]
        mn, mx = np.min(col), np.max(col)
        if mx > mn:
            sc = (mx - mn) / (2**bits - 1)
            result[:, d] = np.round((col - mn) / sc) * sc + mn
        else:
            result[:, d] = col
    # Storage: quantized values + 4 bytes per column for scale/zero (fp32)
    nbytes = (seq * dim * bits) / 8 + dim * 4
    return result, nbytes


def quant_per_token(data: np.ndarray, bits: int) -> Tuple[np.ndarray, float]:
    """
    Per-token (per-row) min/max quantization.
    
    Each token (row) is quantized independently. This is the KIVI strategy for
    value vectors.
    
    Args:
        data: Array of shape (seq_len, dim) to quantize
        bits: Number of bits for quantization
    
    Returns:
        quantized: Dequantized array with same shape as input
        nbytes: Storage cost in bytes
    """
    seq, dim = data.shape
    result = np.zeros_like(data)
    for t in range(seq):
        row = data[t]
        mn, mx = np.min(row), np.max(row)
        if mx > mn:
            sc = (mx - mn) / (2**bits - 1)
            result[t] = np.round((row - mn) / sc) * sc + mn
        else:
            result[t] = row
    # Storage: quantized values + 4 bytes per row for scale/zero (fp32)
    nbytes = (seq * dim * bits) / 8 + seq * 4
    return result, nbytes


def quant_kivi(K: np.ndarray, V: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    KIVI quantization: per-channel for keys, per-token for values.
    
    This is the de-facto industry baseline for KV cache quantization. Compresses
    by ~3.9x at 4-bit but exhibits catastrophic worst-case behavior on outlier
    layers at longer contexts.
    
    Args:
        K: Key tensor of shape (seq_len, head_dim)
        V: Value tensor of shape (seq_len, head_dim)
        bits: Quantization bit width
    
    Returns:
        K_quant, V_quant, total_bytes
    """
    K_q, k_bytes = quant_per_channel(K, bits)
    V_q, v_bytes = quant_per_token(V, bits)
    return K_q, V_q, k_bytes + v_bytes


def norm_direction_quant(
    data: np.ndarray,
    norm_bits: int = 8,
    dir_bits: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    Norm + Direction (N+D) quantization.
    
    Decomposes each vector into:
      - A scalar magnitude (the L2 norm)
      - A unit direction vector (the original / its norm)
    
    Then quantizes each component independently. The magnitude component absorbs
    the outlier scale that would otherwise force per-channel quantization to
    stretch its grid and destroy precision elsewhere. The unit direction vector
    quantizes cleanly because its values are bounded in [-1, 1] by construction.
    
    After quantization, the direction vector is renormalized to compensate for
    quantization error that pushed it slightly off the unit sphere.
    
    Args:
        data: Array of shape (seq_len, dim) to quantize
        norm_bits: Bits for the magnitude scalar (default 8)
        dir_bits: Bits for each direction component (default 4)
    
    Returns:
        reconstructed: Quantized-then-dequantized array, same shape as input
        nbytes: Storage cost in bytes
    
    Recommended configurations:
        N8 + D4: ~3.9x compression, best for quality (avg cos > 0.99, min cos > 0.98)
        N8 + D3: ~5.1x compression, good Pareto point (avg cos ~0.97, min cos > 0.91)
        N8 + D2: ~7.5x compression, aggressive (avg cos ~0.85, use with caution)
    """
    seq, dim = data.shape
    
    # Decompose into magnitude + unit direction
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    directions = data / norms_safe
    
    # Quantize the magnitudes
    if norm_bits >= 16:
        norms_q = norms
        norm_bytes = seq * 2  # fp16
    else:
        mn, mx = np.min(norms), np.max(norms)
        if mx > mn:
            sc = (mx - mn) / (2**norm_bits - 1)
            norms_q = np.round((norms - mn) / sc) * sc + mn
        else:
            norms_q = norms.copy()
        norm_bytes = (seq * norm_bits) / 8 + 4  # values + scale/zero
    
    # Quantize the directions, per-column
    dir_q = np.zeros_like(directions)
    for d in range(dim):
        col = directions[:, d]
        mn, mx = np.min(col), np.max(col)
        if mx > mn:
            sc = (mx - mn) / (2**dir_bits - 1)
            dir_q[:, d] = np.round((col - mn) / sc) * sc + mn
        else:
            dir_q[:, d] = col
    dir_bytes = (seq * dim * dir_bits) / 8 + dim * 4
    
    # Renormalize directions to be unit length (corrects for quantization drift)
    dn = np.linalg.norm(dir_q, axis=1, keepdims=True)
    dn = np.where(dn > 1e-10, dn, 1.0)
    dir_q = dir_q / dn
    
    # Reconstruct
    reconstructed = dir_q * norms_q
    
    return reconstructed, norm_bytes + dir_bytes


def quant_nd_kv(
    K: np.ndarray,
    V: np.ndarray,
    norm_bits: int = 8,
    dir_bits: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply N+D quantization to both keys and values.
    
    Args:
        K: Key tensor (seq_len, head_dim)
        V: Value tensor (seq_len, head_dim)
        norm_bits: Magnitude precision
        dir_bits: Direction precision
    
    Returns:
        K_quant, V_quant, total_bytes
    """
    K_q, k_bytes = norm_direction_quant(K, norm_bits, dir_bits)
    V_q, v_bytes = norm_direction_quant(V, norm_bits, dir_bits)
    return K_q, V_q, k_bytes + v_bytes


def compute_output_cosine(
    K_orig: np.ndarray,
    V_orig: np.ndarray,
    K_comp: np.ndarray,
    V_comp: np.ndarray,
) -> float:
    """
    Measure attention output similarity between original and compressed KV cache.
    
    Computes attention outputs using both the original and compressed caches,
    then returns the mean cosine similarity between the per-token output vectors.
    
    This is the most meaningful quality metric because it measures the effect of
    quantization on what attention actually produces, not just the distance
    between cached vectors.
    
    Args:
        K_orig, V_orig: Original (uncompressed) keys and values
        K_comp, V_comp: Compressed-then-decompressed keys and values
    
    Returns:
        Mean cosine similarity across all token positions (1.0 = perfect, 0.0 = orthogonal)
    """
    seq_len, head_dim = K_orig.shape
    causal = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    
    # Original attention output
    oa = K_orig @ K_orig.T / np.sqrt(head_dim) + causal
    oa = oa - np.max(oa, axis=-1, keepdims=True)
    op = np.exp(oa) / (np.sum(np.exp(oa), axis=-1, keepdims=True) + 1e-10)
    orig_out = op @ V_orig
    
    # Compressed attention output
    ca = K_comp @ K_comp.T / np.sqrt(head_dim) + causal
    ca = ca - np.max(ca, axis=-1, keepdims=True)
    cp = np.exp(ca) / (np.sum(np.exp(ca), axis=-1, keepdims=True) + 1e-10)
    comp_out = cp @ V_comp
    
    # Mean per-token cosine similarity
    norms = np.linalg.norm(orig_out, axis=1) * np.linalg.norm(comp_out, axis=1) + 1e-10
    return float(np.mean(np.sum(orig_out * comp_out, axis=1) / norms))
