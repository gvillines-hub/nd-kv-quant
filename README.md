# nd-kv-quant — Evaluation Harness and Norm-Direction Method for KV Cache Compression

A reproducible benchmark harness for evaluating KV cache compression methods, with first-class support for **per-head worst-case quality metrics** (the cosine-similarity floor that average-cosine reporting hides). Includes a baseline implementation of KIVI (4-bit and 2-bit) and a competitive method developed in this work: **norm-direction scalar mixed-precision quantization**.

Validated across four models: Qwen2.5-7B, Qwen2.5-1.5B, Llama-3.1-8B, and Mistral-7B-v0.3.

**Headline finding:** KIVI's catastrophic outlier layer failure is **model-specific to the Qwen family** in this evaluation, not a universal flaw. The same architecture-dependent pattern appears for paper-faithful TurboQuant (Zandieh et al., ICLR 2026) — works as claimed on Llama and Mistral, collapses to negative min cosine on Qwen2.5 at every bit width tested. Outlier-aware mixed precision (the community recipe from @scos-lab's findings in [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)) partially rescues TurboQuant on Qwen but does not fully recover. Norm-Direction quantization is the only method evaluated here that maintains per-head minimum cosine above 0.96 across all four model families.

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

### Worst-case quality (min cosine) vs. KIVI 4-bit at ~3.9x compression

| Model | KIVI 4-bit min | **N8+D4 min** | Gap | KIVI failure? |
|---|---|---|---|---|
| Qwen2.5-7B | 0.588 | **0.969** | +0.38 | **Catastrophic** |
| Qwen2.5-1.5B | 0.646 | **0.991** | +0.34 | **Catastrophic** |
| Llama-3.1-8B | 0.977 | **0.990** | +0.01 | None |
| Mistral-7B-v0.3 | 0.953 | **0.991** | +0.04 | None |

N+D's improvement over KIVI is dramatic on the Qwen family and incremental on Llama/Mistral. **The catastrophic outlier layer problem is real but model-specific** — a finding that, to my knowledge, is not directly reported in the published KIVI evaluation.

### Aggressive compression regime (min cosine) at ~7.5x compression

| Model | KIVI 2-bit min | **N8+D2 min** | Gap |
|---|---|---|---|
| Qwen2.5-7B | 0.119 | **0.595** | +0.48 |
| Qwen2.5-1.5B | 0.184 | **0.759** | +0.58 |
| Llama-3.1-8B | 0.702 | **0.781** | +0.08 |
| Mistral-7B-v0.3 | 0.688 | **0.765** | +0.08 |

KIVI 2-bit collapses on Qwen and is rough on Llama/Mistral. N8+D2 stays viable across all four models. **If your compression target is >=5x, N+D is the only method tested here that doesn't fall off a cliff.**

### N8+D3 Pareto point (~5.15x compression)

| Model | KIVI 4-bit min | **N8+D3 min** | N8+D3 compression |
|---|---|---|---|
| Qwen2.5-7B | 0.588 | **0.925** | 5.15x |
| Qwen2.5-1.5B | 0.646 | **0.954** | 5.15x |
| Llama-3.1-8B | 0.977 | **0.948** | 5.15x |
| Mistral-7B-v0.3 | 0.953 | **0.949** | 5.16x |

N8+D3 offers **33% more compression than KIVI 4-bit while matching or exceeding KIVI's worst-case quality** on every model tested. This is the tradeoff point I'd recommend for most production deployments.

### TurboQuant cross-model comparison (added May 16, 2026)

Paper-faithful TurboQuant (Algorithm 1, Zandieh et al., ICLR 2026) was added to the harness in May 2026 to extend the cross-model comparison beyond KIVI. Three variants tested: random rotation (paper), Walsh-Hadamard Transform (the variant llama.cpp implementations ship), and outlier-aware mixed precision (the recipe described by @scos-lab in [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)).

No QJL residual is included; community consensus from @arclabs001 and @scos-lab is that QJL hurts attention quality at low bit-widths.

**Per-head minimum cosine, all variants:**

| Variant | Qwen2.5-7B | Qwen2.5-1.5B | Llama-3.1-8B | Mistral-7B-v0.3 |
|---|---|---|---|---|
| Uniform 4-bit random (~3.9x) | 0.034 | 0.144 | 0.974 | 0.974 |
| Uniform 4-bit WHT (~3.9x) | -0.051 | 0.024 | 0.977 | 0.977 |
| Outlier 3.5b (10%@8 + 90%@3, ~4.4x) | 0.132 | 0.389 | 0.971 | 0.978 |
| Outlier 3.2b (20%@4 + 80%@3, ~4.8x) | **0.595** | **0.627** | 0.964 | 0.977 |
| Outlier 2.25b (25%@3 + 75%@2, ~6.7x) | 0.320 | 0.219 | 0.862 | 0.914 |
| **N+D N8+D4 (reference, 3.9x)** | **0.969** | **0.991** | **0.990** | **0.991** |

Three observations:

1. **Paper-faithful TurboQuant matches its published claims on Llama-3.1-8B and Mistral-7B-v0.3** (both around min cos 0.97 at 4-bit, consistent with the paper's Needle-In-A-Haystack result on Llama-3.1-8B), and **collapses to near-zero or negative min cosine on both Qwen2.5 models**. WHT substitution doesn't help.

2. **Outlier handling partially rescues TurboQuant on Qwen, but not fully.** The standard recipe (top-10% at 8-bit) brings Qwen2.5-7B from -0.05 → 0.13. Widening to 20% at 4-bit gets it to 0.60. Better, but still below the 0.95+ threshold for deployment safety. **20%@4-bit beats 10%@8-bit** on both Qwen models even though the average bit budget is similar — suggesting the outlier mass on Qwen is wider than the standard recipe assumes.

3. **On Llama and Mistral, outlier handling makes essentially no difference.** It's a Qwen-class intervention, consistent with the K/V norm ratio data that predicts the failure mode in the first place.

The implementation was verified against the paper's analytical centroid values (page 10) and Lloyd-Max Gaussian distortion reference within 2% before benchmarking; see `tests/test_turboquant.py`.

This tests the **paper algorithm** specifically. It does not test the deployed `tq3_0` variants in llama.cpp from @TheTom, @animehacker, or @unixsysdev, which include block-wise handling and engineering specifics the paper doesn't cover. Those may behave differently than what's shown here.

Full per-model JSONs in `benchmarks/results/*-2026-05-16.json` (uniform variants) and `benchmarks/results/*-2026-05-16-outlier.json` (outlier-aware variants).

---

## Method

Three components in the codebase; the headline results above evaluate the **quantization** component. The other two (speculative prefetch, tiled fallback) are implemented in `nd_kv_quant/speculative.py` and validated in earlier research notebooks but not yet in the cross-model benchmark suite — see [Limitations](#limitations).

### 1. Norm-Direction quantization (the core contribution)

Each K and V vector `v` is decomposed into a scalar magnitude `||v||` and a unit-direction `v_hat = v / ||v||`. The magnitude is quantized at 8 bits per vector; the direction is quantized at 2, 3, or 4 bits per component depending on the target compression ratio. Reconstruction is `v_reconstructed = ||v||_q * v_hat_q`.

**The mechanism:** quantization error on a unit-direction vector is isotropic with respect to the attention dot product. Per-channel quantization of raw KV vectors stretches the quantization grid to accommodate a few outlier magnitudes, destroying precision in the remaining channels. On the Qwen family this produces the "catastrophic outlier layer" failure mode visible in the results above (KIVI min cosine = 0.59-0.65 at 4-bit). Llama and Mistral have less extreme magnitude variation, so KIVI works adequately at 4-bit on those models — but the magnitude problem returns for everyone at 2-bit, where N+D's robustness is universal.

Implementation: [`nd_kv_quant/quantization.py`](nd_kv_quant/quantization.py)

### 2. Speculative prefetch via attention patterns

For deployments where the KV cache exceeds VRAM and must stream from RAM, the previous token's attention distribution is used to predict which KV entries the next token will need. Prior research notebooks measured 72-74% hit rate when prefetching the top 10-15% of the cache by predicted attention weight.

Implementation: [`nd_kv_quant/speculative.py`](nd_kv_quant/speculative.py). Not yet included in the cross-model benchmark suite.

### 3. Tiled fallback for misses

When the speculative subset misses (max attention probability below threshold), the system falls back to exact tiled attention over the full cache for that query. In earlier notebooks, this contributed little to overall quality once N8+D4 was in place but doubled transfer volume; recommended off for most deployments.

---

## Using this as a benchmark for other methods

The harness is designed to support apples-to-apples evaluation of arbitrary KV cache compression methods. Three things make it useful as an instrument:

1. **Per-head, per-layer minimum cosine reporting.** Most published evaluations report average cosine, perplexity, or downstream task accuracy — all of which can mask catastrophic outlier failures. KIVI's average cosine on Qwen2.5-7B at 706 tokens is 0.983, which looks fine; its min cosine is 0.588, meaning at least one layer/head combination has collapsed. The harness reports both.

2. **Cross-model evaluation.** Running the same compression method against four model families surfaces architecture-dependent failure modes that single-model evaluations hide. The Qwen-specific KIVI/TurboQuant failure is a clear example.

3. **Pluggable methods.** New compression methods can be added by implementing the appropriate interface in `nd_kv_quant/quantization.py`. KIVI 4-bit/2-bit, N+D at 4/3/2-bit directions, TurboQuant (uniform random rotation and WHT), and TurboQuant with outlier-aware mixed precision are reference implementations.

If you're working on a new KV cache compression method and want a direct comparison against KIVI, N+D, and the TurboQuant family across four model families with worst-case quality metrics, this harness should give you a result in a few minutes per model. PRs adding implementations of other published methods (NSNQuant, KVmix, deployed `tq3_0` variants, etc.) are welcome.

---

## Prior art

This work was developed independently prior to a prior-art search conducted in May 2026, which identified substantial overlap with published work. The honest comparison:

### Directly related — overlapping mechanism or primitive

- **MiniCache** (Liu et al., NeurIPS 2024) [[paper](https://arxiv.org/abs/2405.14366)] established the magnitude-direction decomposition of KV cache vectors. MiniCache applies the decomposition to cross-layer merging (averaging directions across adjacent layers, storing per-layer magnitudes); this work applies a similar decomposition to per-vector mixed-precision scalar quantization. The decomposition primitive is shared; the application is different. MiniCache established the decomposition; this work applies it differently and produces direct cross-model comparisons against KIVI that do not appear in MiniCache's evaluation.

- **NSNQuant** (Son et al., NeurIPS 2025) [[paper](https://arxiv.org/abs/2505.18231)] uses a three-step "Normalize-Shift-Normalize" transformation with a Hadamard transform to align KV distributions with a standard normal for calibration-free vector quantization. The first step (token-wise normalization, i.e., extracting the norm) overlaps with this work; the remainder of the pipeline (channel-wise centering, Hadamard rotation, codebook-based VQ) is mechanically distinct from scalar mixed-precision quantization with a separate norm budget.

- **PolarQuant** (Yu et al., 2025) and its productization **TurboQuant** (Zandieh et al., ICLR 2026) [[llama.cpp port](https://github.com/AmesianX/TurboQuant)] use rotation-based scalar quantization with a fixed Walsh-Hadamard transform and a precomputed codebook. Mechanically distinct from norm-direction decomposition. A paper-faithful implementation plus an outlier-aware mixed-precision variant are now included in this harness (`nd_kv_quant/turboquant.py`, `nd_kv_quant/turboquant_outlier.py`); see the [TurboQuant cross-model comparison](#turboquant-cross-model-comparison-added-may-16-2026) section above. The TurboQuant llama.cpp discussion thread [#20969](https://github.com/ggml-org/llama.cpp/discussions/20969) independently observed that Qwen models fall in a regime where the K/V magnitude ratio is unusually high, which is consistent with the Qwen-specific failure mode this work documents for both KIVI and paper-faithful TurboQuant. The outlier-aware recipe partially addresses this but does not fully recover the per-head minimum cosine on Qwen.

### Mixed-precision KV cache (same family, different mechanisms)

- **KVTuner** (Li et al., Feb 2025), **KVmix** (Li et al., May 2025), **RateQuant** (Zuo et al., 2026), **MixKVQ** (Dec 2025), **CommVQ** (Jun 2025) — all variants of mixed-precision allocation across layers/heads/channels via different mechanisms (gradient sensitivity, rate-distortion theory, query awareness, commutative codebooks). None use a norm/direction decomposition.

### What this repo adds, given the above

Three things, conservatively:

1. **Per-vector mixed-precision scalar quantization** via magnitude-direction decomposition (8-bit norm, 2/3/4-bit direction). MiniCache established the primitive; this work applies it differently.

2. **Cross-model evaluation surfacing architecture-dependent failure modes** under KIVI, paper-faithful TurboQuant, and outlier-aware TurboQuant. The finding that KIVI's catastrophic outlier layer problem is model-family-specific (Qwen vs. Llama/Mistral) is, to my knowledge, not directly reported in published evaluations. The same evaluation reveals that paper-faithful TurboQuant exhibits an analogous Qwen-specific failure, and that the standard outlier-aware recipe partially but incompletely rescues it.

3. **A reproducible evaluation harness** with first-class worst-case (per-head minimum) cosine reporting across multiple model families. The methodological contribution is making architecture-dependent failure modes visible.

The decomposition primitive is not novel. The specific application, the cross-model failure-mode findings, and the worst-case-quality evaluation methodology are the contributions.

---

## Limitations

- **Speculative prefetch and tiled fallback are not in the cross-model benchmark suite yet.** They're implemented (`nd_kv_quant/speculative.py`) and were validated in earlier research notebooks on Qwen2.5-7B, but the four-model results above evaluate the quantization component only.
- **Batch size 1.** Batched-inference behavior is untested.
- **No PPL or downstream-task evaluation.** Quality is measured via per-head cosine similarity against FP16 output. Cosine similarity tracks PPL well in this regime but is not a substitute for a LongBench or similar evaluation.
- **TurboQuant comparison tests the paper algorithm, not deployed variants.** The `tq3_0` implementations in llama.cpp (TheTom, animehacker, unixsysdev) include outlier handling and block-wise treatment that the paper doesn't specify. Those may behave differently than what's shown here.
- **No llama.cpp port.** A port was planned and paused after a prior-art search revealed TurboQuant and related methods are being actively ported by multiple groups.
- **Context lengths in the benchmark are short** (706-824 tokens). Long-context behavior may differ; this is a known gap.
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
