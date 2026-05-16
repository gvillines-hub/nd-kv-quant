"""
Helpers for extracting KV cache tensors from HuggingFace transformer models.

These functions load a model, run a forward pass on a prompt, and extract the
resulting KV cache as NumPy arrays organized by layer and head. The quantization
and speculative-attention code in this package operates on these per-head
tensors.

GPU is required to run real models. The extracted KV cache is moved to CPU and
returned as NumPy arrays so the rest of the pipeline can run without GPU memory
pressure.
"""

import numpy as np
from typing import List, Tuple


def extract_kv_cache(
    model_name: str = "Qwen/Qwen2.5-7B",
    prompt: str = None,
    use_4bit: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int, int]:
    """
    Load a model, run forward pass, extract KV cache as NumPy arrays.
    
    Args:
        model_name: HuggingFace model identifier
        prompt: Text prompt for KV extraction. If None, uses a long default prompt.
        use_4bit: Load model in 4-bit (requires bitsandbytes). If False, uses fp16.
    
    Returns:
        all_K: List of per-layer key tensors, each shape (n_heads, seq_len, head_dim)
        all_V: List of per-layer value tensors, same shape as all_K
        n_layers: Number of layers
        n_heads: Number of KV heads (may differ from query heads for GQA models)
        seq_len: Sequence length of the prompt after tokenization
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "extract_kv_cache requires torch and transformers. "
            "Install with: pip install torch transformers accelerate"
        ) from e
    
    if prompt is None:
        prompt = _default_prompt()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=qconfig, device_map={"": 0}
            )
        except ImportError:
            print("bitsandbytes not available, falling back to fp16")
            use_4bit = False
    
    if not use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map={"": 0}
        )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    kv = outputs.past_key_values
    n_layers = len(kv)
    n_heads = kv[0][0].shape[1]
    head_dim = kv[0][0].shape[3]
    
    all_K = [kv[l][0][0].float().cpu().numpy() for l in range(n_layers)]
    all_V = [kv[l][1][0].float().cpu().numpy() for l in range(n_layers)]
    
    # Free GPU memory
    del outputs, kv, model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return all_K, all_V, n_layers, n_heads, seq_len


def compute_ground_truth_attention(
    all_K: List[np.ndarray],
    all_V: List[np.ndarray],
    n_layers: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
) -> Tuple[dict, dict, np.ndarray]:
    """
    Pre-compute ground-truth attention outputs and probabilities for every (layer, head).
    
    These are used as reference outputs to measure quality of compressed/approximated
    attention. Pre-computing once avoids redundant work during benchmarking.
    
    Returns:
        ground_truth: dict mapping (layer, head) -> attention output (seq_len, head_dim)
        all_probs: dict mapping (layer, head) -> attention probabilities (seq_len, seq_len)
        token_importance: array (n_layers, n_heads, seq_len) summing attention into each token
    """
    ground_truth = {}
    all_probs = {}
    token_importance = np.zeros((n_layers, n_heads, seq_len))
    causal = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    
    for l in range(n_layers):
        for h in range(n_heads):
            K = all_K[l][h]
            V = all_V[l][h]
            attn = K @ K.T / np.sqrt(head_dim) + causal
            attn = attn - np.max(attn, axis=-1, keepdims=True)
            probs = np.exp(attn) / (np.sum(np.exp(attn), axis=-1, keepdims=True) + 1e-10)
            all_probs[(l, h)] = probs
            ground_truth[(l, h)] = probs @ V
            token_importance[l, h] = np.sum(probs, axis=0)
    
    return ground_truth, all_probs, token_importance


def _default_prompt() -> str:
    """Default prompt used for KV cache extraction in benchmarks."""
    return """The United States workforce development system operates through a complex
network of local workforce development boards that administer federal programs under the
Workforce Innovation and Opportunity Act. These boards are responsible for oversight of
training providers, eligibility determination, and performance accountability across
multi-county regions throughout the nation.

Performance measurement in workforce development relies on quarterly wage record matching
with state unemployment insurance databases. The system tracks participant outcomes including
employment rates in the second and fourth quarters after exit, median earnings, credential
attainment rates, and measurable skill gains during program participation.

The mathematical foundations of neural network compression share surprising parallels with
signal processing theory developed during the Cold War era. Signals intelligence operations
required compressing vast amounts of intercepted communications into actionable intelligence
summaries. The key insight was that most signals contain redundant information that can be
predicted from context, and only the unpredictable residuals need to be stored or transmitted.

Modern large language models face an analogous challenge. During inference, the key-value
cache grows linearly with sequence length, consuming memory proportional to the number of
layers times the number of attention heads times the sequence length times the head dimension.
For a seventy billion parameter model processing a thirty-two thousand token context window,
this cache alone requires approximately twenty gigabytes of memory in half precision format.

The attention mechanism computes similarity scores between query vectors and all cached key
vectors, then uses these scores to create weighted combinations of cached value vectors.
This operation has quadratic complexity in sequence length, making it the primary bottleneck
for long-context applications. Various approaches have been proposed to address this,
including sparse attention patterns, linear attention approximations, and hierarchical
attention mechanisms that operate at multiple resolution levels.

Quantization reduces numerical precision from sixteen-bit floating point to lower bit widths.
Four-bit quantization achieves approximately four times compression but introduces rounding
errors that accumulate across layers. The distribution of values matters enormously: normally
distributed values quantize well because most values fall near the center of the range, while
distributions with heavy tails or outliers lose significant information when extreme values
are clipped or when the quantization grid is stretched to accommodate them.

An alternative perspective comes from information geometry, which views probability
distributions as points on a Riemannian manifold. The Fisher information metric defines a
natural distance between distributions, and geodesics on this manifold represent the most
efficient paths between statistical states. If the key-value cache vectors lie on or near
a low-dimensional manifold embedded in the high-dimensional vector space, then compression
can be achieved by learning the manifold structure and encoding positions on it rather than
storing the full vectors.

Hyperbolic geometry offers another promising framework. Unlike Euclidean space, hyperbolic
space has exponentially growing volume relative to radius, making it naturally suited to
representing hierarchical structures such as parse trees, taxonomies, and organizational
hierarchies. Language inherently contains hierarchical structure at multiple levels.

The Doppler effect in physics describes how the observed frequency of a wave changes when
the source and observer are in relative motion. An analogous effect may exist in transformer
attention: as the model processes tokens sequentially, the effective frequency content of
key vectors may shift systematically across layers.

Gravitational models provide yet another analogy. In an attention mechanism, each query
vector is attracted to key vectors in proportion to their similarity, much like gravitational
attraction between masses. High-attention tokens act like massive bodies that capture the
attention orbits of many queries."""
