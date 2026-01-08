#!/bin/bash
# Run all experiments for Euclidean Approximate Attention

set -e

echo "========================================"
echo "Euclidean Approximate Attention"
echo "Experiment Runner"
echo "========================================"

# Create results directory
mkdir -p results

# Run tests first
echo ""
echo "[0/3] Running tests..."
python tests/test_basic.py

# Experiment 1: Distance approximation
echo ""
echo "[1/3] Running distance approximation experiment..."
python experiments/01_distance_approximation.py

# Experiment 2: Attention patterns
echo ""
echo "[2/3] Running attention pattern analysis..."
python experiments/02_attention_patterns.py

# Experiment 3: Compute benchmarks
echo ""
echo "[3/3] Running compute benchmarks..."
python experiments/03_compute_benchmarks.py

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Results saved to: ./results/"
echo "========================================"

