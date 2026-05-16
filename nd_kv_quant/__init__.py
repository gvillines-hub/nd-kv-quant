"""
nd-kv-quant: Norm+Direction quantization for transformer KV cache.

A simple, practical implementation of magnitude-direction decomposed quantization
for transformer key-value cache compression. Builds on MiniCache (NeurIPS 2024)
and PolarQuant (2025); uses a simpler per-column quantization of the unit direction
vector instead of recursive polar transformation.

Quick start:
    from nd_kv_quant import norm_direction_quant, compute_output_cosine
    K_quant, n_bytes = norm_direction_quant(K, norm_bits=8, dir_bits=4)
"""

from .quantization import (
    norm_direction_quant,
    quant_nd_kv,
    quant_kivi,
    quant_per_channel,
    quant_per_token,
    compute_output_cosine,
)

from .speculative import (
    speculative_nd_attention,
    full_pipeline,
)

__version__ = "0.1.0"

__all__ = [
    "norm_direction_quant",
    "quant_nd_kv",
    "quant_kivi",
    "quant_per_channel",
    "quant_per_token",
    "compute_output_cosine",
    "speculative_nd_attention",
    "full_pipeline",
]
