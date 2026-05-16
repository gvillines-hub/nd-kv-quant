"""
Outlier-aware TurboQuant extension.

Implements the outlier-channel mixed-precision recipe described by @scos-lab
in the llama.cpp TurboQuant discussion thread #20969:

  > Outlier-Aware Mixed Precision
  > ~5-20% of K channels (especially Layer 0) have 10-100x larger RMS than
  > median. Storing outlier channels at 8-bit, rest at 3-bit:
  >
  > Method                     Avg bits   PPL change (Qwen2.5-1.5B)
  > Uniform K=6, V=3           4.5        +78.1%
  > Mixed K=3, V=3 (outliers)  3.6         +2.1%

The mechanism:
  1. Per-call, compute per-channel RMS on the input data
  2. Identify the top-K channels by RMS as "outliers"
  3. Pull outlier channels out of the vector entirely, quantize them with
     simple per-channel scalar min-max at higher precision (preserves them)
  4. Apply paper-faithful TurboQuant (rotation + Lloyd-Max codebook) to the
     remaining non-outlier subspace at lower precision
  5. Reconstruct by combining the two subspaces back at their original positions

The non-outlier subspace uses a random orthogonal rotation matrix sized for its
dimension. WHT is not supported in the outlier-aware variant because the
non-outlier dimension count is rarely a power of 2.
"""

import numpy as np
from typing import Tuple

from nd_kv_quant.turboquant import (
    get_random_rotation,
    get_turboquant_codebook,
    _quantize_to_codebook,
)


def _identify_outlier_channels(data: np.ndarray, fraction: float) -> np.ndarray:
    """
    Identify the top fraction of channels by per-channel RMS.

    Args:
        data: array of shape (seq_len, dim)
        fraction: fraction of channels to mark as outliers (0.0 to 1.0)

    Returns:
        sorted 1D array of outlier channel indices, length round(dim * fraction)
    """
    seq, dim = data.shape
    rms = np.sqrt(np.mean(data ** 2, axis=0))
    n_outliers = max(1, int(round(dim * fraction)))
    outlier_idx = np.argsort(rms)[-n_outliers:]
    outlier_idx.sort()
    return outlier_idx


def _scalar_minmax_quantize(data: np.ndarray, bits: int) -> np.ndarray:
    """
    Per-channel min-max scalar quantization. Used for outlier channels.

    Args:
        data: array of shape (seq_len, n_channels)
        bits: number of bits per value

    Returns:
        Dequantized array of same shape.
    """
    seq, n = data.shape
    result = np.zeros_like(data)
    for c in range(n):
        col = data[:, c]
        mn, mx = float(np.min(col)), float(np.max(col))
        if mx > mn:
            sc = (mx - mn) / (2 ** bits - 1)
            result[:, c] = np.round((col - mn) / sc) * sc + mn
        else:
            result[:, c] = col
    return result


def turboquant_data_outlier_aware(
    data: np.ndarray,
    bits_outlier: int = 8,
    bits_normal: int = 3,
    outlier_fraction: float = 0.10,
    norm_bits: int = 16,
) -> Tuple[np.ndarray, float]:
    """
    Outlier-aware TurboQuant.

    Args:
        data: array of shape (seq_len, dim) to quantize
        bits_outlier: bits per value for outlier channels (default 8)
        bits_normal: bits per coordinate for non-outlier channels after rotation (default 3)
        outlier_fraction: fraction of channels treated as outliers (default 0.10)
        norm_bits: bits for the per-vector L2 norm on the non-outlier subspace (default 16 = fp16)

    Returns:
        reconstructed: dequantized array of same shape as input
        nbytes: per-cache storage cost
    """
    seq, dim = data.shape

    # 1. Identify outlier channels by RMS
    outlier_idx = _identify_outlier_channels(data, outlier_fraction)
    n_outlier = len(outlier_idx)
    n_normal = dim - n_outlier
    normal_idx = np.setdiff1d(np.arange(dim), outlier_idx)

    # 2. Quantize outlier channels with scalar min-max at high precision
    outlier_data = data[:, outlier_idx]
    outlier_q = _scalar_minmax_quantize(outlier_data, bits_outlier)
    outlier_bytes = (seq * n_outlier * bits_outlier) / 8 + n_outlier * 4

    # 3. Quantize non-outlier subspace with TurboQuant (random rotation + codebook)
    normal_data = data[:, normal_idx]
    R_normal = get_random_rotation(n_normal, seed=42 + n_normal)

    norms = np.linalg.norm(normal_data, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    unit_normal = normal_data / norms_safe

    if norm_bits >= 16:
        norms_q = norms
        norm_bytes = seq * 2  # fp16 per token
    else:
        mn, mx = float(np.min(norms)), float(np.max(norms))
        if mx > mn:
            sc = (mx - mn) / (2 ** norm_bits - 1)
            norms_q = np.round((norms - mn) / sc) * sc + mn
        else:
            norms_q = norms.copy()
        norm_bytes = (seq * norm_bits) / 8 + 4

    rotated = unit_normal @ R_normal.T
    codebook = get_turboquant_codebook(bits_normal, n_normal)
    rotated_q = _quantize_to_codebook(rotated, codebook)
    reconstructed_unit = rotated_q @ R_normal
    normal_q = reconstructed_unit * norms_q

    normal_bytes = (seq * n_normal * bits_normal) / 8 + norm_bytes

    # 4. Reassemble: place outlier and non-outlier columns back at original positions
    result = np.zeros_like(data)
    result[:, outlier_idx] = outlier_q
    result[:, normal_idx] = normal_q

    return result, outlier_bytes + normal_bytes


def quant_turboquant_outlier_aware_kv(
    K: np.ndarray,
    V: np.ndarray,
    bits_outlier: int = 8,
    bits_normal: int = 3,
    outlier_fraction: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply outlier-aware TurboQuant to both K and V (separate outlier
    identification per tensor, since K and V have very different outlier structure).
    """
    K_q, k_bytes = turboquant_data_outlier_aware(
        K, bits_outlier=bits_outlier, bits_normal=bits_normal,
        outlier_fraction=outlier_fraction
    )
    V_q, v_bytes = turboquant_data_outlier_aware(
        V, bits_outlier=bits_outlier, bits_normal=bits_normal,
        outlier_fraction=outlier_fraction
    )
    return K_q, V_q, k_bytes + v_bytes
