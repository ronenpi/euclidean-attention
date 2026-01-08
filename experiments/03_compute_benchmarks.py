"""
Experiment 3: Compute Benchmarks

Goal: Demonstrate quadratic → linear scaling for attention computation.

Measurements:
- Wall-clock time vs sequence length
- Memory usage
- FLOPs (theoretical)

Compare:
- Standard O(n²) attention
- Exact Euclidean O(n²) attention
- Approximate Euclidean O(n*k) attention
"""
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import gc

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from attention import StandardAttention, EuclideanAttention, ApproximateEuclideanAttention


@dataclass
class BenchmarkConfig:
    embed_dim: int = 64
    num_heads: int = 4
    batch_size: int = 1
    seq_lengths: List[int] = None
    num_landmarks: int = 32
    num_warmup: int = 3
    num_trials: int = 10
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]


def get_memory_usage(device: str) -> float:
    """Get current memory usage in MB."""
    if device == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    elif device == 'mps':
        # MPS doesn't have direct memory query, return 0
        return 0.0
    else:
        return 0.0


def benchmark_attention(
    attn_module: nn.Module,
    X: torch.Tensor,
    num_warmup: int,
    num_trials: int,
    device: str,
) -> Dict[str, float]:
    """Benchmark a single attention module."""
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = attn_module(X)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Clear memory
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Measure memory before
    mem_before = get_memory_usage(device)
    
    # Timing trials
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        with torch.no_grad():
            output = attn_module(X)
        
        if device == 'cuda':
            torch.cuda.synchronize()
            
        end = time.perf_counter()
        times.append(end - start)
    
    # Measure memory after (peak during forward)
    mem_after = get_memory_usage(device)
    
    return {
        'mean_time': np.mean(times) * 1000,  # Convert to ms
        'std_time': np.std(times) * 1000,
        'min_time': np.min(times) * 1000,
        'max_time': np.max(times) * 1000,
        'memory_mb': max(0, mem_after - mem_before),
    }


def theoretical_flops(
    method: str,
    n: int,
    d: int,
    h: int,
    k: int,
) -> int:
    """Compute theoretical FLOPs for each method."""
    head_dim = d // h
    
    if method == 'standard':
        # Q, K, V projections: 3 * n * d * d
        proj = 3 * n * d * d
        # QK^T: n * n * head_dim * h
        qk = n * n * head_dim * h * 2
        # Softmax: ~5 * n * n * h
        softmax = 5 * n * n * h
        # Attention @ V: n * n * head_dim * h
        av = n * n * head_dim * h * 2
        # Output proj: n * d * d
        out = n * d * d
        return proj + qk + softmax + av + out
    
    elif method == 'euclidean':
        # Same as standard (distance computation is similar FLOPs to dot product)
        return theoretical_flops('standard', n, d, h, k)
    
    elif method == 'approximate':
        # Projections: 3 * n * d * d
        proj = 3 * n * d * d
        # Distance to landmarks: n * k * head_dim * h
        dist = n * k * head_dim * h * 2
        # W inverse: k^3 * h
        inv = k ** 3 * h
        # Z = Phi_K.T @ V: k * n * head_dim * h
        z = k * n * head_dim * h * 2
        # Y = W^-1 @ Z: k * k * head_dim * h
        y = k * k * head_dim * h * 2
        # Output = Phi_Q @ Y: n * k * head_dim * h
        out_attn = n * k * head_dim * h * 2
        # Output proj: n * d * d
        out = n * d * d
        return proj + dist + inv + z + y + out_attn + out
    
    return 0


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Dict]:
    """Run benchmarks for all methods across sequence lengths."""
    results = {
        'standard_O(n²)': {},
        'euclidean_exact_O(n²)': {},
        'euclidean_approx_O(nk)': {},
    }
    
    print(f"Benchmarking on device: {config.device}")
    print(f"Embed dim: {config.embed_dim}, Heads: {config.num_heads}")
    print(f"Batch size: {config.batch_size}, Landmarks: {config.num_landmarks}")
    print("-" * 60)
    
    for n in tqdm(config.seq_lengths, desc="Sequence lengths"):
        # Generate input
        X = torch.randn(
            config.batch_size, n, config.embed_dim,
            device=config.device
        )
        
        # Initialize modules
        std_attn = StandardAttention(
            config.embed_dim, config.num_heads
        ).to(config.device)
        
        euc_attn = EuclideanAttention(
            config.embed_dim, config.num_heads
        ).to(config.device)
        
        approx_attn = ApproximateEuclideanAttention(
            config.embed_dim, config.num_heads,
            num_landmarks=config.num_landmarks
        ).to(config.device)
        
        # Skip if sequence too short for landmarks
        if n < config.num_landmarks:
            continue
        
        # Benchmark each method
        try:
            results['standard_O(n²)'][n] = benchmark_attention(
                std_attn, X, config.num_warmup, config.num_trials, config.device
            )
            results['standard_O(n²)'][n]['flops'] = theoretical_flops(
                'standard', n, config.embed_dim, config.num_heads, config.num_landmarks
            )
        except RuntimeError as e:
            print(f"Standard attention failed at n={n}: {e}")
            results['standard_O(n²)'][n] = None
        
        try:
            results['euclidean_exact_O(n²)'][n] = benchmark_attention(
                euc_attn, X, config.num_warmup, config.num_trials, config.device
            )
            results['euclidean_exact_O(n²)'][n]['flops'] = theoretical_flops(
                'euclidean', n, config.embed_dim, config.num_heads, config.num_landmarks
            )
        except RuntimeError as e:
            print(f"Euclidean attention failed at n={n}: {e}")
            results['euclidean_exact_O(n²)'][n] = None
            
        try:
            results['euclidean_approx_O(nk)'][n] = benchmark_attention(
                approx_attn, X, config.num_warmup, config.num_trials, config.device
            )
            results['euclidean_approx_O(nk)'][n]['flops'] = theoretical_flops(
                'approximate', n, config.embed_dim, config.num_heads, config.num_landmarks
            )
        except RuntimeError as e:
            print(f"Approximate attention failed at n={n}: {e}")
            results['euclidean_approx_O(nk)'][n] = None
        
        # Clean up
        del X, std_attn, euc_attn, approx_attn
        gc.collect()
        if config.device == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def plot_scaling(results: Dict, config: BenchmarkConfig, save_path: Path):
    """Plot scaling curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {
        'standard_O(n²)': '#e74c3c',
        'euclidean_exact_O(n²)': '#3498db', 
        'euclidean_approx_O(nk)': '#2ecc71'
    }
    
    # Time vs sequence length
    ax = axes[0]
    for method, data in results.items():
        ns = sorted([n for n in data.keys() if data[n] is not None])
        times = [data[n]['mean_time'] for n in ns]
        stds = [data[n]['std_time'] for n in ns]
        
        ax.errorbar(ns, times, yerr=stds, fmt='o-', 
                   label=method, color=colors[method], capsize=3)
    
    ax.set_xlabel('Sequence Length (n)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Wall-Clock Time vs Sequence Length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Theoretical FLOPs
    ax = axes[1]
    for method, data in results.items():
        ns = sorted([n for n in data.keys() if data[n] is not None])
        flops = [data[n]['flops'] for n in ns]
        
        ax.plot(ns, flops, 'o-', label=method, color=colors[method])
    
    ax.set_xlabel('Sequence Length (n)')
    ax.set_ylabel('FLOPs')
    ax.set_title('Theoretical FLOPs vs Sequence Length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speedup vs standard
    ax = axes[2]
    ns_common = sorted(set(results['standard_O(n²)'].keys()) & 
                       set(results['euclidean_approx_O(nk)'].keys()))
    ns_common = [n for n in ns_common if results['standard_O(n²)'][n] and results['euclidean_approx_O(nk)'][n]]
    
    if ns_common:
        # Theoretical FLOPs ratio (this shows the real scaling benefit)
        flops_ratios = [
            results['standard_O(n²)'][n]['flops'] / results['euclidean_approx_O(nk)'][n]['flops']
            for n in ns_common
        ]
        
        ax.plot(ns_common, flops_ratios, 'o-', color='#9b59b6', linewidth=2, markersize=8)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(ns_common, 1, flops_ratios, alpha=0.3, color='#9b59b6')
    
    ax.set_xlabel('Sequence Length (n)')
    ax.set_ylabel('FLOPs Reduction (O(n²) / O(nk))')
    ax.set_title(f'Theoretical Speedup of Linear Attention (k={config.num_landmarks})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_results_table(results: Dict, config: BenchmarkConfig):
    """Print results as a table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    
    ns = sorted(set().union(*[d.keys() for d in results.values()]))
    
    header = f"{'n':<8}"
    for method in results.keys():
        header += f" {method:<28}"
    print(header)
    print("-" * 100)
    
    for n in ns:
        row = f"{n:<8}"
        for method, data in results.items():
            if n in data and data[n] is not None:
                time_str = f"{data[n]['mean_time']:.2f}±{data[n]['std_time']:.2f}ms"
                row += f" {time_str:<28}"
            else:
                row += f" {'N/A':<28}"
        print(row)
    
    print("\n" + "-" * 100)
    print("SPEEDUP (Linear O(nk) vs Quadratic O(n²)):")
    print("-" * 100)
    
    for n in ns:
        if (n in results['standard_O(n²)'] and results['standard_O(n²)'][n] and
            n in results['euclidean_approx_O(nk)'] and results['euclidean_approx_O(nk)'][n]):
            speedup = results['standard_O(n²)'][n]['mean_time'] / results['euclidean_approx_O(nk)'][n]['mean_time']
            flops_ratio = results['standard_O(n²)'][n]['flops'] / results['euclidean_approx_O(nk)'][n]['flops']
            print(f"n={n:<6}: {speedup:.2f}x wall-clock, {flops_ratio:.1f}x theoretical FLOPs reduction")


def main():
    print("\n" + "=" * 60)
    print("COMPUTE BENCHMARKS")
    print("=" * 60 + "\n")
    
    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'benchmarks'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration - test at larger scales to see crossover
    config = BenchmarkConfig(
        embed_dim=64,
        num_heads=4,
        batch_size=1,
        seq_lengths=[256, 512, 1024, 2048, 4096, 8192],
        num_landmarks=64,  # More landmarks for better quality
        num_warmup=3,
        num_trials=5,  # Fewer trials for faster results
        device=device,
    )
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    results = run_benchmark(config)
    
    # Print results
    print_results_table(results, config)
    
    # Plot
    plot_scaling(results, config, results_dir / 'scaling_benchmark.png')
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

