# Research Plan: Euclidean Approximate Attention with Linear-Time Distance Estimation

## Overview

**Thesis**: Linear-time distance approximations can produce useful attention patterns, improve scalability, and enable longer contexts.

**Foundation**: [Indyk et al., 2019 - Sample-Optimal Low-Rank Approximation of Distance Matrices](https://arxiv.org/abs/1906.00339)

Key result from the paper:
> A rank-k approximation of an n×m distance matrix can be computed by reading only **O((n+m)k/ε)** entries, in **Õ(n+m)·poly(k,1/ε)** time.

This is **sublinear** in the matrix size, which is impossible for general matrices!

---

## Research Questions

### Primary Questions

1. **Can we use linear-time distance estimation to build efficient attention?**
   - Replace O(n²) dot-product attention with O(n·k) Euclidean attention

2. **What is the quality/speed tradeoff?**
   - How much approximation error can we tolerate?
   - What rank k is needed for good attention patterns?

3. **Does Euclidean attention have different properties than dot-product?**
   - Different inductive biases
   - Different gradient flow
   - Different behavior at scale

### Secondary Questions

4. How does this compare to existing linear attention methods?
5. Can pre-trained models be adapted to use Euclidean attention?
6. What are the best landmark selection strategies?

---

## Theoretical Background

### Standard Attention
```
Attention(Q, K, V) = softmax(QK^T / √d) @ V
```
- Complexity: O(n² · d)
- Memory: O(n²)

### Euclidean Attention
```
Attention(Q, K, V) = softmax(-||Q - K||² / τ) @ V
```

Key relationship:
```
||q - k||² = ||q||² + ||k||² - 2⟨q, k⟩
```

This means Euclidean attention is **sensitive to vector norms**, not just angles.

### Approximate Euclidean Attention

Using Nyström approximation with k landmarks:

1. Compute C_Q: distances from queries to landmarks — O(n·k·d)
2. Compute C_K: distances from keys to landmarks — O(n·k·d)  
3. Compute W: distances among landmarks — O(k²·d)
4. Kernel approximation: K ≈ Φ_Q @ W^{-1} @ Φ_K^T

**Linear attention trick**:
```
output = softmax(K) @ V 
       ≈ normalize(Φ_Q @ (W^{-1} @ (Φ_K^T @ V)))
```

This is **O(n·k·d + k³)** instead of O(n²·d)!

---

## Experimental Plan

### Phase 1: Distance Approximation Quality (Week 1-2)

**Goal**: Validate that we can approximate distance matrices accurately.

**Experiments**:
- `01_distance_approximation.py`
- Test low-rank, JL, and Nyström approximations
- Metrics: MSE, relative Frobenius error, Spearman correlation

**Expected Outcome**: 
- With k = O(√n), achieve < 10% relative error
- Error decreases predictably with rank

### Phase 2: Attention Pattern Analysis (Week 2-3)

**Goal**: Understand how Euclidean attention differs from standard.

**Experiments**:
- `02_attention_patterns.py`
- Visualize attention patterns
- Measure: entropy, sparsity, locality

**Expected Outcomes**:
- Euclidean attention naturally more local (nearby = smaller distance)
- Approximate attention preserves structure with low error

### Phase 3: Compute Benchmarks (Week 3-4)

**Goal**: Demonstrate O(n²) → O(n·k) scaling.

**Experiments**:
- `03_compute_benchmarks.py`
- Wall-clock time vs sequence length
- Memory profiling

**Expected Outcomes**:
- At n > 1000, approximate attention is faster
- Memory usage grows linearly vs quadratically

### Phase 4: Small Transformer Training (Week 4-6)

**Goal**: Train small models to validate end-to-end quality.

**Experiments** (to be added):
- Train tiny GPT-style models (6-12 layers, ~10M params)
- Dataset: WikiText-103 or small subset
- Compare perplexity between attention types

**Baselines**:
- Standard attention (FlashAttention if available)
- Performer (random feature linear attention)
- Linear Transformer

### Phase 5: Long Context Evaluation (Week 6-8)

**Goal**: Show advantage at longer sequences.

**Experiments** (to be added):
- Passkey retrieval test
- Needle-in-haystack test  
- Context length scaling: 1K → 8K → 32K

---

## Implementation Status

### Completed ✅

- [x] Exact Euclidean distance computation
- [x] Low-rank distance approximation (Nyström-style)
- [x] Johnson-Lindenstrauss projection
- [x] Standard attention baseline
- [x] Euclidean attention (exact)
- [x] Approximate Euclidean attention
- [x] Basic tests
- [x] Experiment 1: Distance approximation
- [x] Experiment 2: Attention patterns
- [x] Experiment 3: Compute benchmarks

### TODO 📋

- [ ] Run full experiments and collect results
- [ ] Implement transformer blocks with Euclidean attention
- [ ] Small model training experiments
- [ ] Long context experiments
- [ ] Comparison with Performer, Linear Transformer
- [ ] Write-up and analysis

---

## Hardware Requirements

**Development**: Apple Silicon (MPS) - works for small experiments

**Training**: Google Colab with 2× NVIDIA GPUs
- Needed for Phase 4+ (model training)
- Recommend A100 or T4 instances

---

## Key Metrics to Track

1. **Distance Approximation**
   - Relative Frobenius error
   - Spearman correlation (rank preservation)

2. **Attention Quality**
   - Cosine similarity of outputs vs exact
   - Attention pattern correlation

3. **Compute Efficiency**
   - Wall-clock time vs sequence length
   - Memory usage
   - Speedup ratio

4. **Model Quality**
   - Perplexity on validation set
   - Downstream task accuracy

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Approximation error too high | Tune k, ε; try adaptive landmarks |
| Training instability | Gradient clipping, careful initialization |
| No speedup in practice | Profile and optimize; try different k |
| Worse than baselines | Document negative result; analyze why |

---

## Timeline (Flexible)

| Week | Focus |
|------|-------|
| 1-2 | Phase 1: Distance approximation experiments |
| 2-3 | Phase 2: Attention pattern analysis |
| 3-4 | Phase 3: Compute benchmarks |
| 4-6 | Phase 4: Small model training |
| 6-8 | Phase 5: Long context + write-up |

---

## References

1. Indyk, P., Vakilian, A., Wagner, T., & Woodruff, D. P. (2019). *Sample-Optimal Low-Rank Approximation of Distance Matrices*. NeurIPS 2019.

2. Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS 2017.

3. Choromanski, K., et al. (2020). *Rethinking Attention with Performers*. ICLR 2021.

4. Katharopoulos, A., et al. (2020). *Transformers are RNNs*. ICML 2020.

