# Euclidean Approximate Attention with Linear-Time Distance Estimation

## Hypothesis

Linear-time distance approximations (based on [Indyk et al., 2019](https://arxiv.org/abs/1906.00339)) can produce useful attention patterns, improve scalability, and enable longer contexts.

## Core Idea

Standard attention computes `softmax(QK^T/√d)` — a dot-product similarity in O(n²) time.

We propose **Euclidean Attention**: `softmax(-D²/τ)` where D is the pairwise distance matrix.

**Key insight from the paper**: Distance matrices have special structure allowing:
- Read only O((n+m)k/ε) entries (sublinear!)
- Compute rank-k approximation in Õ(n+m) time
- Achieve error bound: ||A - VU||²_F ≤ ||A - A_k||²_F + ε||A||²_F

## Project Goals

1. **Distance approximation quality**: How close are estimated distances to true distances?
2. **Compute reduction**: Show quadratic → linear attention cost
3. **Model quality**: Compare downstream accuracy to standard attention
4. **Long context**: Demonstrate better behavior at longer sequence lengths
5. **Practicality**: Plug into existing models with minimal changes
6. **Baselines**: Compare against modern linear-attention methods

## Setup

```bash
cd euclidean-attention
pip install -r requirements.txt
```

## Experiments

```bash
# Phase 1: Distance approximation quality
python experiments/01_distance_approximation.py

# Phase 2: Attention pattern analysis  
python experiments/02_attention_patterns.py

# Phase 3: Small model training
python experiments/03_small_transformer.py

# Phase 4: Scaling benchmarks
python experiments/04_scaling_benchmarks.py
```

## Hardware

- Local: Apple Silicon (MPS)
- Cloud: Google Colab with 2× NVIDIA GPUs

## References

- Indyk, P., Vakilian, A., Wagner, T., & Woodruff, D. P. (2019). 
  *Sample-Optimal Low-Rank Approximation of Distance Matrices*. NeurIPS 2019.

