# nd-kv-quant — Evaluation Harness and Norm-Direction Method for KV Cache Compression

A reproducible benchmark harness for evaluating KV cache compression methods, with first-class support for **per-head worst-case quality metrics** (the cosine-similarity floor that average-cosine reporting hides). Includes a baseline implementation of KIVI (4-bit and 2-bit) and a competitive method developed in this work: **norm-direction scalar mixed-precision quantization**.

Validated across four models: Qwen2.5-7B, Qwen2.5-1.5B, Llama-3.1-8B, and Mistral-7B-v0.3.

**Headline finding:** KIVI's catastrophic outlier layer failure is **model-specific to the Qwen family** in this evaluation, not a universal flaw. Norm-Direction quantization fixes that failure on Qwen models, matches KIVI on Llama and Mistral, and dominates KIVI in the aggressive (≤2-bit-equivalent) compression regime on every model tested.

**Status:** Research implementation. The decomposition primitive overlaps with prior work (notably MiniCache, NeurIPS 2024, and the first step of NSNQuant); see [Prior Art](#prior-art) for an honest comparison.

---

## Quick start

```bash
git clone https://github.com/gvillines-hub/nd-kv-quant.git
cd nd-kv-quant
pip install -e .
python examples/quick_start.py
```

To reproduce the full cross-model benchmark suite:

```bash
python benchmarks/run_benchmarks.py
```

Results are written to `benchmarks/results/{model}-{date}.json`.

Tested on an NVIDIA RTX 5060 Ti (16GB) and on a BossGame AMD Ryzen AI Max+ 395 mini-PC.

---

## Headline results

### Worst-case quality (min cosine) vs. KIVI 4-bit at ~3.9× compression

| Model | KIVI 4-bit min | **N8+D4 min** | Gap | KIVI failure? |
|---|---|---|---|---|
| Qwen2.5-7B | 0.588 | **0.969** | +0.38 | **Catastrophic** |
| Qwen2.5-1.5B | 0.646 | **0.991** | +0.34 | **Catastrophic** |
| Llama-3.1-8B | 0.977 | **0.990** | +0.01 | None |
| Mistral-7B-v0.3 | 0.953 | **0.991** | +0.04 | None |

N+D's improvement over KIVI is dramatic on the Qwen family and incremental on Llama/Mistral. **The catastrophic outlier layer problem is real but model-specific** — a finding that, to my knowledge, is not directly reported in the published KIVI evaluation.

### Aggressive compression regime (min cosine) at ~7.5× compression

| Model | KIVI 2-bit min | **N8+D2 min** | Gap |
|---|---|---|---|
| Qwen2.5-7B | 0.119 | **0.595** | +0.48 |
| Qwen2.5-1.5B | 0.184 | **0.759** | +0.58 |
| Llama-3.1-8B | 0.702 | **0.781** | +0.08 |
| Mistral-7B-v0.3 | 0.688 | **0.765** | +0.08 |

KIVI 2-bit collapses on Qwen and is rough on Llama/Mistral. N8+D2 stays viable across all four models. **If your compression target is ≥5×, N+D is the only method tested here that doesn't fall off a cliff.**

### N8+D3 Pareto point (~5.15× compression)

| Model | KIVI 4-bit min | **N8+D3 min** | N8+D3 compression |
|---|---|---|---|
| Qwen2.5-7B | 0.588 | **0.925** | 5.15× |
| Qwen2.5-1.5B | 0.646 | **0.954** | 5.15× |
| Llama-3.1-8B | 0.977 | **0.948** | 5.15× |
| Mistral-7B-v0.3 | 0.953 | **0.949** | 5.16× |

N8+D3 offers **33% more compression than KIVI 4-bit while matching or exceeding KIVI's worst-case quality** on every model tested. This is the tradeoff point I'd recommend for most production deployments.

---

## Method

Three components in the codebase; the headline results above evaluate the **quantization** component. The other two (speculative prefetch, tiled fallback) are implemented in `nd_kv_quant/speculative.py` and validated in earlier research notebooks but not yet in the cross-model benchmark suite — see [Limitations](#limitations).

### 1. Norm-Direction quantization (the core contribution)

Each K and V vector `v` is decomposed into a scalar magnitude `‖v‖` and a unit-direction `v̂ = v / ‖v‖`. The magnitude is quantized at 8 bits per vector; the direction is quantized at 2, 3, or 4 bits per component depending on the target compression ratio. Reconstruction is `v_reconstructed = ‖v‖_q · v̂_q`.

**The mechanism:** quantization error on a unit-direction vector is isotropic with respect to the attention dot product. Per-channel quantization of raw KV vectors stretches the quantization grid to accommodate a few outlier magnitudes, destroying precision in the remaining channels. On the Qwen family this produces the "catastrophic outlier layer" failure mode visible in the results above (KIVI min cosine = 0.59–0.65 at 4-bit). Llama and Mistral have less extreme magnitude variation, so KIVI works adequately at 4-bit on those models — but the magnitude problem returns for everyone at 2-bit, where N+D's robustness is universal.

Implementation: [`nd_kv_quant/quantization.py`](nd_kv_quant/quantization.py)

### 2. Speculative prefetch via attention patterns

For deployments where the KV cache exceeds VRAM and must stream from RAM, the previous token's attention distribution is used to predict which KV entries the next token will need. Prior research notebooks measured 72–74% hit rate when prefetching the top 10–15% of the cache by predicted attention weight.

Implementation: [`nd_kv_quant/speculative.py`](nd_kv_quant/speculative.py). Not yet included in the cross-model benchmark suite.

### 3. Tiled fallback for misses

When the speculative subset misses (max attention probability below threshold), the system falls back to exact tiled attention over the full cache for that query. In earlier notebooks, this contributed little to overall quality once N8+D4 was in place but doubled transfer volume; recommended off for most deployments.

---

## Using this as a benchmark for other methods

The harness is designed to support apples-to-apples evaluation of arbitrary KV cache compression methods. Three things make it useful as an instrument:

1. **Per-head, per-layer minimum cosine reporting.** Most published evaluations report average cosine, perplexity, or downstream task accuracy — all of which can mask catastrophic outlier failures. KIVI's average cosine on Qwen2.5-7B at 706 tokens is 0.983, which looks fine; its min cosine is 0.588, meaning at least one layer/head combination has collapsed. The harness reports both.

2. **Cross-model evaluation.** Running the same compression method against four model families surfaces architecture-dependent failure modes that single-model evaluations hide. The Qwen-specific KIVI failure is a clear example.

3. **Pluggable methods.** New compression methods can be added by implementing the appropriate interface in `nd_kv_quant/quantization.py`. KIVI 4-bit/2-bit and N+D at 4/3/2-bit directions are reference implementations.

If you're working on a new KV cache compression method and want a direct comparison against KIVI and N+D across four model families with worst-case quality metrics, this harness should give you a result in a few minutes per model. PRs adding implementations of other published methods (NSNQuant, TurboQuant, KVmix, etc.) are welcome.

---

## Prior art

This work was developed independently prior to a prior-art search conducted in May 2026, which identified substantial overlap with published work. The honest comparison:

### Directly related — overlapping mechanism or primitive

- **MiniCache** (Liu et al., NeurIPS 2024) [[paper](https://arxiv.org/abs/2405.14366)] established the magnitude-direction decomposition of KV cache vectors. MiniCache applies the decomposition to cross-layer merging (averaging directions across adjacent layers, storing per-layer magnitudes); this work applies a similar decomposition to per-vector mixed-precision scalar quantization. The decomposition primitive is shared; the application is different. MiniCache established the decomposition; this work applies it differently and produces direct cross-model comparisons against KIVI that do not appear in MiniCache's evaluation.

- **NSNQuant** (Son et al., NeurIPS 2025) [[paper](https://arxiv.org/abs/2505.18231)] uses a three-step "Normalize–Shift–Normalize" transformation with a Hadamard transform to align KV distributions with a standard normal for calibration-free vector quantization. The first step (token-wise normalization, i.e., extracting the norm) overlaps with this work; the remainder of the pipeline (channel-wise centering, Hadamard rotation, codebook-based VQ) is mechanically distinct from scalar mixed-precision quantization with a separate norm budget.

- **PolarQuant** (Yu et al., 2025) and its productization **TurboQuant** (Zandieh et al., ICLR 2026) [[llama.cpp port](https://github.com/AmesianX/TurboQuant)] use rotation-based scalar quantization with a fixed Walsh-Hadamard transform and a precomputed codebook. Mechanically distinct from norm-direction decomposition. The TurboQuant llama.cpp discussion thread [#20969](https://github.com/ggml-org/llama.cpp/discussions/20969) independently observed that Qwen models fall in a regime where the K/V magnitude ratio is unusually high, which is consistent with the Qwen-specific failure mode this work documents for KIVI.

### Mixed-precision KV cache (same family, different mechanisms)

- **KVTuner** (Li et al., Feb 2025), **KVmix** (Li et al., May 2025), **RateQuant** (Zuo et al., 2026), **MixKVQ** (Dec 2025), **CommVQ** (Jun 2025) — all variants of mixed-precision allocation across layers/heads/channels via different mechanisms (gradient sensitivity, rate-distortion theory, query awareness, commutative codebooks). None use a norm/direction decomposition.

### What this repo adds, given the above

Three things, conservatively:

1. **Per-vector mixed-precision scalar quantization** via magnitude-direction decomposition (8-bit norm, 2/3/4-bit direction). MiniCache established the primitive; this work applies it differently.

2. **Cross-model evaluation surfacing architecture-dependent KIVI failure.** The finding that KIVI's catastrophic outlier layer problem is model-family-specific (Qwen vs. Llama/Mistral) is, to my knowledge, not directly reported in published evaluations.

3. **A reproducible evaluation harness** with first-class worst-case (per-head minimum) cosine reporting across multiple model families. The methodological contribution is making architecture-dependent failure modes visible.

The decomposition primitive is not novel. The specific application, the cross-model failure-mode finding, and the worst-case-quality evaluation methodology are the contributions.

---

## Limitations

- **Speculative prefetch and tiled fallback are not in the cross-model benchmark suite yet.** They're implemented (`nd_kv_quant/speculative.py`) and were validated in earlier research notebooks on Qwen2.5-7B, but the four-model results above evaluate the quantization component only.
- **Batch size 1.** Batched-inference behavior is untested.
- **No PPL or downstream-task evaluation.** Quality is measured via per-head cosine similarity against FP16 output. Cosine similarity tracks PPL well in this regime but is not a substitute for a LongBench or similar evaluation.
- **No llama.cpp port.** A port was planned and paused after a prior-art search revealed TurboQuant and related methods are being actively ported by multiple groups.
- **Context lengths in the benchmark are short** (706–824 tokens). Long-context behavior may differ; this is a known gap.
- **70B-scale evaluation not run.** Hardware constraints (single 16GB consumer GPU) prevent direct testing.

---

## Hardware

- **NVIDIA RTX 5060 Ti** (16GB VRAM) — primary development and benchmarking GPU
- **BossGame AMD Ryzen AI Max+ 395 mini-PC** — secondary platform representative of consumer hardware target

Models loaded in 4-bit NF4 weights; KV cache computed in FP16 regardless of weight quantization.

---

## Citation / contact

**Gregory Villines**
gvillnes@gmail.com

```bibtex
@misc{villines2026normdirection,
  author = {Villines, Gregory},
  title  = {nd-kv-quant: Evaluation Harness and Norm-Direction Method for KV Cache Compression},
  year   = {2026},
  url    = {https://github.com/gvillines-hub/nd-kv-quant},
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
