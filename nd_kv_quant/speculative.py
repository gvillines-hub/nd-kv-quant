"""
Speculative prefetch for KV cache streaming.

Uses the previous token's attention distribution to predict which cached
tokens the next query will attend to most heavily, prefetches that subset
from RAM to GPU, and computes attention on the prefetched subset.

Optional tiled fallback: if the prefetched subset's max attention probability
falls below a confidence threshold, fall back to full (exact) attention over
all tokens. The fallback is mathematically exact at any tile size; speculative
mode is the fast path.

Related published work:
  - SpeCache (Liu et al., ICML 2025): uses speculative-token-based prediction
  - InfiniGen (OSDI 2024): uses previous-layer attention for prediction
  - AttentionPredictor (2025): learns temporal patterns in attention scores

This implementation uses the previous-token attention pattern directly, which
is simpler than the published alternatives. The implementation here is a
reference for evaluation purposes; production use would need integration with
specific inference engines (llama.cpp, vLLM, etc.).
"""

import numpy as np
from typing import Tuple
from .quantization import norm_direction_quant


def speculative_nd_attention(
    K: np.ndarray,
    V: np.ndarray,
    probs_full: np.ndarray,
    prefetch_frac: float = 0.10,
    norm_bits: int = 8,
    dir_bits: int = 4,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Run attention with N+D quantization plus speculative prefetch.
    
    The KV cache is stored in (simulated) RAM as N+D quantized data. For each
    query position, only the top-k tokens predicted by the previous token's
    attention pattern are "transferred" to GPU and used for attention.
    
    Args:
        K: Key tensor (seq_len, head_dim) - the full ground-truth cache
        V: Value tensor (seq_len, head_dim)
        probs_full: Ground-truth attention probabilities (seq_len, seq_len)
        prefetch_frac: Fraction of tokens to prefetch (0.10 = 10%)
        norm_bits: N+D magnitude precision
        dir_bits: N+D direction precision
    
    Returns:
        output: Attention output (seq_len, head_dim)
        ram_compression: Storage compression vs FP16
        hit_rate: Fraction of actually-important tokens that were prefetched
        xfer_ratio: Average fraction of cache transferred per query
    """
    seq, dim = K.shape
    
    # Quantize once (this lives in RAM)
    K_q, k_bytes = norm_direction_quant(K, norm_bits, dir_bits)
    V_q, v_bytes = norm_direction_quant(V, norm_bits, dir_bits)
    fp16_bytes = seq * dim * 2 * 2  # K + V at fp16
    ram_compression = fp16_bytes / (k_bytes + v_bytes)
    
    output = np.zeros((seq, dim))
    
    # First token: nothing to predict from, transfer just the initial token
    output[0] = V_q[0]
    
    hits = 0
    misses = 0
    tokens_transferred = [1]
    
    for q in range(1, seq):
        # Predict from previous token's attention pattern
        k_fetch = max(1, int(q * prefetch_frac))
        pf_idx = set(np.argsort(probs_full[q - 1, :q])[-k_fetch:])
        
        # Always include current, previous, and first tokens for stability
        pf_idx.add(q)
        pf_idx.add(q - 1)
        pf_idx.add(0)
        pf_idx = sorted(pf_idx)
        
        tokens_transferred.append(len(pf_idx))
        
        # Hit rate measurement: did we prefetch the tokens that actually mattered?
        actual_top = set(np.argsort(probs_full[q, :q + 1])[-k_fetch:])
        hits += len(actual_top & set(pf_idx))
        misses += len(actual_top - set(pf_idx))
        
        # Compute attention on prefetched subset
        pi = np.array(pf_idx)
        K_pf = K_q[pi]
        V_pf = V_q[pi]
        
        scores = K_pf @ K_q[q] / np.sqrt(dim)
        scores = scores - np.max(scores)
        probs_pf = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-10)
        output[q] = probs_pf @ V_pf
    
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    xfer_ratio = float(np.mean(tokens_transferred)) / seq
    
    return output, ram_compression, hit_rate, xfer_ratio


def full_pipeline(
    K: np.ndarray,
    V: np.ndarray,
    probs_full: np.ndarray,
    prefetch_frac: float = 0.10,
    norm_bits: int = 8,
    dir_bits: int = 4,
    miss_threshold: float = 0.5,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Full deployment pipeline: speculative prefetch with tiled fallback.
    
    For each query, runs speculative attention on the prefetched subset. If the
    maximum attention probability in the prefetched subset is below the miss
    threshold, falls back to exact attention over the full (quantized) cache.
    
    Args:
        K, V: Full uncompressed keys and values
        probs_full: Ground-truth attention probabilities
        prefetch_frac: Fraction to prefetch on the fast path
        norm_bits, dir_bits: N+D parameters
        miss_threshold: Confidence below which we fall back to full attention.
                        Set to 0.0 to always use speculative; 1.0 to always tile.
    
    Returns:
        output: Attention output
        spec_pct: Percentage of queries handled via speculative fast path
        tile_pct: Percentage that fell back to full attention
        avg_tokens: Average tokens transferred per query
    """
    seq, dim = K.shape
    K_q, _ = norm_direction_quant(K, norm_bits, dir_bits)
    V_q, _ = norm_direction_quant(V, norm_bits, dir_bits)
    
    output = np.zeros((seq, dim))
    output[0] = V_q[0]
    
    spec_used = 1
    tile_used = 0
    tokens_total = 1
    
    for q in range(1, seq):
        k_fetch = max(1, int(q * prefetch_frac))
        pf_idx = set(np.argsort(probs_full[q - 1, :q])[-k_fetch:])
        pf_idx.add(q)
        pf_idx.add(q - 1)
        pf_idx.add(0)
        pf_idx = sorted(pf_idx)
        
        pi = np.array(pf_idx)
        scores = K_q[pi] @ K_q[q] / np.sqrt(dim)
        scores = scores - np.max(scores)
        probs_pf = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-10)
        
        if np.max(probs_pf) >= miss_threshold or len(pf_idx) > q * 0.5:
            # Fast path: speculative attention on prefetched subset
            output[q] = probs_pf @ V_q[pi]
            spec_used += 1
            tokens_total += len(pf_idx)
        else:
            # Fallback: exact attention on all tokens up to q
            K_full = K_q[:q + 1]
            V_full = V_q[:q + 1]
            full_scores = K_full @ K_q[q] / np.sqrt(dim)
            full_scores = full_scores - np.max(full_scores)
            full_probs = np.exp(full_scores) / (np.sum(np.exp(full_scores)) + 1e-10)
            output[q] = full_probs @ V_full
            tile_used += 1
            tokens_total += (q + 1)
    
    spec_pct = spec_used / seq * 100
    tile_pct = tile_used / seq * 100
    avg_tokens = tokens_total / seq
    
    return output, spec_pct, tile_pct, avg_tokens
