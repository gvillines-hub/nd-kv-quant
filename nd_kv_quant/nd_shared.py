"""
N+D variant with a shared empirical Lloyd-Max codebook on direction components.

This is an ablation, not a deployment method. Tests the hypothesis:
is per-column min-max adaptation doing meaningful work on anisotropic models,
or would a single empirical Lloyd-Max codebook over all direction components
give equivalent quality at lower metadata overhead?

The codebook is derived empirically per-call: collect all direction values from
the (seq_len * dim) sample, fit Lloyd-Max via the iterative algorithm, and
quantize every direction value with nearest-neighbor lookup.

If shared codebook matches per-column min-max in quality, per-column metadata
(4 bytes per channel) is unnecessary overhead. If shared codebook underperforms
significantly on anisotropic models, per-column adaptation is doing real work
and should be preserved.
"""

import numpy as np
from typing import Tuple


def _lloyd_max_empirical(samples: np.ndarray, bits: int, n_iters: int = 50) -> np.ndarray:
    """
    Solve Lloyd-Max on an empirical distribution given by samples.

    Args:
        samples: 1D array of observed values
        bits: codebook size is 2^bits
        n_iters: Lloyd-Max iterations (convergence is typically fast)

    Returns:
        Sorted array of 2^bits centroids.
    """
    K = 2 ** bits
    samples = np.asarray(samples).ravel()

    # Initialize centroids at uniform quantiles
    quantiles = np.linspace(0.5 / K, 1 - 0.5 / K, K)
    centroids = np.quantile(samples, quantiles)

    for _ in range(n_iters):
        # Boundaries: midpoints between consecutive centroids
        boundaries = np.concatenate([
            [-np.inf],
            0.5 * (centroids[:-1] + centroids[1:]),
            [np.inf],
        ])

        bucket_idx = np.searchsorted(boundaries, samples) - 1
        bucket_idx = np.clip(bucket_idx, 0, K - 1)

        new_centroids = np.zeros(K)
        for i in range(K):
            mask = bucket_idx == i
            if mask.sum() > 0:
                new_centroids[i] = samples[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(new_centroids, centroids, atol=1e-9):
            break
        centroids = new_centroids

    centroids.sort()
    return centroids


def _quantize_to_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Map each value to the nearest centroid in a sorted codebook."""
    idx = np.searchsorted(codebook, values)
    idx = np.clip(idx, 1, len(codebook) - 1)
    left = codebook[idx - 1]
    right = codebook[idx]
    take_left = np.abs(values - left) <= np.abs(values - right)
    return np.where(take_left, left, right)


def norm_direction_quant_shared(
    data: np.ndarray,
    norm_bits: int = 8,
    dir_bits: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    N+D with a shared empirical Lloyd-Max codebook on direction components.

    Same magnitude handling as the original N+D, but the unit-direction
    vector is quantized with a single codebook derived empirically from all
    direction values in the (seq_len * dim) sample, instead of per-column
    min-max.

    Args:
        data: array of shape (seq_len, dim) to quantize
        norm_bits: bits for the magnitude scalar (default 8)
        dir_bits: bits per direction component (default 4)

    Returns:
        reconstructed: dequantized array of same shape as input
        nbytes: per-cache storage cost
    """
    seq, dim = data.shape

    # Decompose into magnitude + unit direction (same as N+D)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    directions = data / norms_safe

    # Quantize magnitudes (same as N+D)
    if norm_bits >= 16:
        norms_q = norms
        norm_bytes = seq * 2
    else:
        mn, mx = float(np.min(norms)), float(np.max(norms))
        if mx > mn:
            sc = (mx - mn) / (2 ** norm_bits - 1)
            norms_q = np.round((norms - mn) / sc) * sc + mn
        else:
            norms_q = norms.copy()
        norm_bytes = (seq * norm_bits) / 8 + 4

    # Build a single Lloyd-Max codebook over all direction values
    codebook = _lloyd_max_empirical(directions.ravel(), dir_bits)

    # Quantize all direction values with the shared codebook
    dir_q = _quantize_to_codebook(directions, codebook)

    # Renormalize directions (compensates for quantization drift)
    dn = np.linalg.norm(dir_q, axis=1, keepdims=True)
    dn = np.where(dn > 1e-10, dn, 1.0)
    dir_q = dir_q / dn

    # Storage: direction indices + codebook (2^dir_bits fp32 centroids)
    codebook_bytes = (2 ** dir_bits) * 4
    dir_bytes = (seq * dim * dir_bits) / 8 + codebook_bytes

    reconstructed = dir_q * norms_q
    return reconstructed, norm_bytes + dir_bytes


def quant_nd_kv_shared(
    K: np.ndarray,
    V: np.ndarray,
    norm_bits: int = 8,
    dir_bits: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply N+D-shared-codebook to both K and V."""
    K_q, k_bytes = norm_direction_quant_shared(K, norm_bits, dir_bits)
    V_q, v_bytes = norm_direction_quant_shared(V, norm_bits, dir_bits)
    return K_q, V_q, k_bytes + v_bytes
