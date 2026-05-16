"""
TurboQuant: rotation-based vector quantization for KV cache compression.

Implements TurboQuant_MSE from Zandieh et al. (Google Research, ICLR 2026):
  https://arxiv.org/abs/2504.19874

The method works by:
  1. Rotating each vector into a basis where coordinates are approximately
     independent and identically distributed (Beta on [-1, 1]). The paper uses
     a random orthogonal rotation; llama.cpp implementations (TheTom, animehacker,
     unixsysdev) substitute a Walsh-Hadamard Transform for speed.
  2. Quantizing each coordinate of the rotated vector with a shared Lloyd-Max
     scalar codebook designed for the Beta distribution. Because coordinates
     become i.i.d. after rotation, the same codebook serves all dimensions.
  3. Storing the L2 norm separately at fp16, dequantizing by reversing the
     rotation and scaling by the stored norm.

Two variants are implemented:
  - TurboQuant-RandomRot (Algorithm 1 in the paper)
  - TurboQuant-WHT       (the variant llama.cpp implementations actually ship)

The QJL residual extension (TurboQuant_prod, Algorithm 2 in the paper) is not
implemented. Community findings (Arclabs001/YATQ, scos-lab) show QJL hurts
attention quality at low bit-widths because softmax tolerates uniform bias but
amplifies per-token variance. MSE-only is consistently better in the regimes
of interest for KV cache.
"""

import numpy as np
from typing import Tuple
from scipy.special import gammaln
from scipy.integrate import quad
from scipy.linalg import hadamard


# ---------------------------------------------------------------------------
# Beta distribution on [-1, 1] for coordinates of a randomly-rotated unit vector
# (Lemma 1 from the paper).
# ---------------------------------------------------------------------------

def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    PDF of a single coordinate of a uniformly-random unit vector in R^d.
    
    fX(x) = Γ(d/2) / (sqrt(π) * Γ((d-1)/2)) * (1 - x²)^((d-3)/2)
    
    Uses log-gamma to avoid overflow for large d.
    """
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    norm = np.exp(log_norm)
    # Clip x to (-1, 1) to avoid 0^negative at endpoints for d <= 3
    xc = np.clip(x, -0.999999, 0.999999)
    return norm * (1 - xc**2) ** ((d - 3) / 2)


def _lloyd_max_beta(bits: int, d: int, n_iters: int = 100) -> np.ndarray:
    """
    Solve Lloyd-Max scalar quantization on the Beta distribution from Lemma 1.
    
    Returns an array of 2^bits centroids in [-1, 1] that minimize MSE distortion
    for samples drawn from fX(x; d).
    
    Args:
        bits: Number of bits (codebook size is 2^bits)
        d: Vector dimension (for the Beta shape parameter)
        n_iters: Lloyd-Max iterations (converges fast; 100 is overkill)
    
    Returns:
        Sorted array of centroids, shape (2^bits,)
    """
    K = 2 ** bits
    
    # Initialize centroids at uniform quantiles of the distribution.
    # For symmetric Beta around 0, this works well.
    quantiles = np.linspace(0.5 / K, 1 - 0.5 / K, K)
    # Inverse-CDF sampling via numerical integration would be ideal, but
    # since fX is symmetric and approximately N(0, 1/d) for large d, we use
    # that as the initialization.
    centroids = np.array([
        np.sqrt(1 / d) * _normal_ppf(q) for q in quantiles
    ])
    # Clip into [-1, 1]
    centroids = np.clip(centroids, -0.999, 0.999)
    centroids.sort()
    
    for _ in range(n_iters):
        # Boundaries: midpoints between consecutive centroids
        boundaries = np.concatenate([
            [-1.0],
            0.5 * (centroids[:-1] + centroids[1:]),
            [1.0],
        ])
        
        new_centroids = np.zeros(K)
        for i in range(K):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi <= lo:
                new_centroids[i] = centroids[i]
                continue
            # E[X | X in [lo, hi]] = ∫ x f(x) dx / ∫ f(x) dx
            num, _ = quad(lambda x: x * _beta_pdf(x, d), lo, hi, limit=100)
            den, _ = quad(lambda x: _beta_pdf(x, d), lo, hi, limit=100)
            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = centroids[i]
        
        if np.allclose(new_centroids, centroids, atol=1e-9):
            break
        centroids = new_centroids
    
    centroids.sort()
    return centroids


def _normal_ppf(q: float) -> float:
    """Standard normal inverse-CDF (probit). Used for centroid initialization."""
    # Beasley-Springer-Moro approximation, accurate enough for initialization.
    if q < 0.5:
        return -_normal_ppf(1 - q)
    t = np.sqrt(-2 * np.log(1 - q))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1] * t + c[2] * t**2) / (1 + d[0] * t + d[1] * t**2 + d[2] * t**3)


# ---------------------------------------------------------------------------
# Codebook cache: solving Lloyd-Max takes a few seconds, so memoize per (bits, d).
# ---------------------------------------------------------------------------

_CODEBOOK_CACHE: dict = {}

def get_turboquant_codebook(bits: int, d: int) -> np.ndarray:
    """Return cached Lloyd-Max codebook for (bits, d), computing if needed."""
    key = (bits, d)
    if key not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[key] = _lloyd_max_beta(bits, d)
    return _CODEBOOK_CACHE[key]


# ---------------------------------------------------------------------------
# Rotations.
# ---------------------------------------------------------------------------

_ROTATION_CACHE: dict = {}

def get_random_rotation(d: int, seed: int = 42) -> np.ndarray:
    """
    Random orthogonal matrix via QR decomposition of a Gaussian matrix.
    
    Deterministic given the seed. Same dimension always produces the same matrix
    so results are reproducible across runs.
    """
    key = ("rand", d, seed)
    if key not in _ROTATION_CACHE:
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(G)
        # Fix sign convention so QR is canonical
        Q = Q * np.sign(np.diag(R))
        _ROTATION_CACHE[key] = Q
    return _ROTATION_CACHE[key]


def get_wht_matrix(d: int) -> np.ndarray:
    """
    Normalized Walsh-Hadamard Transform matrix of size d x d.
    
    Requires d to be a power of 2. For non-power-of-2 head dimensions you would
    typically pad or use a block-Hadamard variant; the four models in our suite
    (Qwen2.5-7B, Qwen2.5-1.5B, Llama-3.1-8B, Mistral-7B-v0.3) all have head_dim=128
    which is a clean power of 2.
    """
    key = ("wht", d)
    if key not in _ROTATION_CACHE:
        if (d & (d - 1)) != 0:
            raise ValueError(f"WHT requires d to be a power of 2; got d={d}")
        H = hadamard(d).astype(np.float64) / np.sqrt(d)
        _ROTATION_CACHE[key] = H
    return _ROTATION_CACHE[key]


# ---------------------------------------------------------------------------
# Core quantization function.
# ---------------------------------------------------------------------------

def _quantize_to_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    Map each value to the nearest centroid in the codebook.
    
    Args:
        values: array of any shape
        codebook: sorted 1D array of centroids
    
    Returns:
        Array of same shape as values, each entry replaced by nearest codebook entry.
    """
    # Use searchsorted to find candidate insertion points, then pick nearest neighbor.
    idx = np.searchsorted(codebook, values)
    idx = np.clip(idx, 1, len(codebook) - 1)
    left = codebook[idx - 1]
    right = codebook[idx]
    take_left = np.abs(values - left) <= np.abs(values - right)
    return np.where(take_left, left, right)


def turboquant_data(
    data: np.ndarray,
    bits: int,
    rotation: str = "random",
    norm_bits: int = 16,
) -> Tuple[np.ndarray, float]:
    """
    Apply TurboQuant_MSE to a (seq_len, dim) tensor.
    
    Args:
        data: Array of shape (seq_len, dim) to quantize
        bits: Bits per coordinate after rotation (codebook size 2^bits)
        rotation: "random" for QR-based random orthogonal (paper Algorithm 1),
                  "wht" for Walsh-Hadamard Transform (the variant llama.cpp ships)
        norm_bits: Bits for the per-vector L2 norm (default 16 = fp16, stored verbatim)
    
    Returns:
        reconstructed: Quantized-then-dequantized array, same shape as input
        nbytes: Per-cache storage cost (excludes rotation matrix and codebook,
                which are model-level overhead amortized across all tokens)
    """
    seq, dim = data.shape
    
    # Pick rotation
    if rotation == "random":
        R = get_random_rotation(dim)
    elif rotation == "wht":
        R = get_wht_matrix(dim)
    else:
        raise ValueError(f"unknown rotation '{rotation}'; expected 'random' or 'wht'")
    
    # Store L2 norms separately (paper: "compute and store the L2 norms in floating-point
    # precision and rescale the dequantized points using these stored norms").
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    unit = data / norms_safe
    
    # Optionally quantize norms (default: keep at fp16)
    if norm_bits >= 16:
        norms_q = norms
        norm_bytes = seq * 2  # fp16 per token
    else:
        # Min-max scalar quant for norms, matching the convention in N+D
        mn, mx = float(np.min(norms)), float(np.max(norms))
        if mx > mn:
            sc = (mx - mn) / (2 ** norm_bits - 1)
            norms_q = np.round((norms - mn) / sc) * sc + mn
        else:
            norms_q = norms.copy()
        norm_bytes = (seq * norm_bits) / 8 + 4  # values + scale/zero
    
    # Rotate into the "almost-Beta-distributed" basis
    rotated = unit @ R.T   # shape (seq, dim), each row is R @ unit_row
    
    # Quantize each coordinate with the shared codebook
    codebook = get_turboquant_codebook(bits, dim)
    rotated_q = _quantize_to_codebook(rotated, codebook)
    
    # Rotate back, then rescale by the original norm
    reconstructed_unit = rotated_q @ R   # since R is orthogonal, R.T inverse = R
    reconstructed = reconstructed_unit * norms_q
    
    # Storage cost:
    #   - indices: seq * dim * bits / 8  bytes
    #   - norms: norm_bytes
    # (Codebook 2^bits * 4 bytes and rotation matrix dim^2 * 4 bytes are model-level
    # overhead, not per-cache. Same convention as the paper.)
    index_bytes = (seq * dim * bits) / 8
    return reconstructed, index_bytes + norm_bytes


# ---------------------------------------------------------------------------
# KV wrappers matching the convention in quantization.py.
# ---------------------------------------------------------------------------

def quant_turboquant_kv(
    K: np.ndarray,
    V: np.ndarray,
    bits: int,
    rotation: str = "random",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply TurboQuant to both keys and values with the same bit width and rotation type.
    
    Args:
        K: Key tensor (seq_len, head_dim)
        V: Value tensor (seq_len, head_dim)
        bits: Bits per coordinate (typically 2, 3, or 4)
        rotation: "random" (paper) or "wht" (llama.cpp variant)
    
    Returns:
        K_quant, V_quant, total_bytes
    """
    K_q, k_bytes = turboquant_data(K, bits, rotation=rotation)
    V_q, v_bytes = turboquant_data(V, bits, rotation=rotation)
    return K_q, V_q, k_bytes + v_bytes
