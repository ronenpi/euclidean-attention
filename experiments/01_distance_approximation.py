"""
Experiment 1: Distance Approximation Quality

Goal: Evaluate how well our distance estimators approximate true Euclidean distances.

Metrics:
- Mean Squared Error (MSE)
- Relative Frobenius Error: ||D - D_approx||_F / ||D||_F
- Spearman Correlation (rank preservation)
- Max Absolute Error

Test conditions:
- Varying sequence lengths (n)
- Varying dimensions (d)
- Varying approximation rank (k)
- Different data distributions (uniform, Gaussian, real embeddings)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from distance_estimators import (
    compute_squared_euclidean_distance_matrix,
    LowRankDistanceApproximator,
    JLDistanceEstimator,
    NystromDistanceEstimator,
)


@dataclass
class ExperimentConfig:
    """Configuration for distance approximation experiment."""
    seq_lengths: List[int] = None
    dimensions: List[int] = None
    ranks: List[int] = None
    epsilon: float = 0.1
    num_trials: int = 5
    seed: int = 42
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [64, 128, 256, 512, 1024]
        if self.dimensions is None:
            self.dimensions = [32, 64, 128, 256]
        if self.ranks is None:
            self.ranks = [8, 16, 32, 64, 128]


def generate_synthetic_data(
    n: int, 
    d: int, 
    distribution: str = 'gaussian',
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate synthetic embedding data."""
    if distribution == 'gaussian':
        X = torch.randn(n, d, device=device)
    elif distribution == 'uniform':
        X = torch.rand(n, d, device=device) * 2 - 1  # [-1, 1]
    elif distribution == 'clustered':
        # Data clustered around k centers
        k = min(10, n // 10)
        centers = torch.randn(k, d, device=device) * 3
        assignments = torch.randint(0, k, (n,))
        X = centers[assignments] + torch.randn(n, d, device=device) * 0.5
    elif distribution == 'transformer_like':
        # Simulate transformer hidden states (normalized, structured)
        X = torch.randn(n, d, device=device)
        X = X / X.norm(dim=-1, keepdim=True)  # Normalize
        X = X * (1 + 0.3 * torch.randn(n, 1, device=device))  # Vary norms slightly
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return X


def compute_metrics(D_true: torch.Tensor, D_approx: torch.Tensor) -> Dict[str, float]:
    """Compute approximation quality metrics."""
    # MSE
    mse = ((D_true - D_approx) ** 2).mean().item()
    
    # Relative Frobenius error
    rel_frob = torch.norm(D_true - D_approx, 'fro') / torch.norm(D_true, 'fro')
    rel_frob = rel_frob.item()
    
    # Max absolute error
    max_abs = (D_true - D_approx).abs().max().item()
    
    # Spearman correlation (rank preservation)
    D_true_flat = D_true.flatten().cpu().numpy()
    D_approx_flat = D_approx.flatten().cpu().numpy()
    
    # Compute ranks
    true_ranks = D_true_flat.argsort().argsort()
    approx_ranks = D_approx_flat.argsort().argsort()
    
    # Spearman correlation
    n = len(true_ranks)
    d_ranks = true_ranks - approx_ranks
    spearman = 1 - 6 * (d_ranks ** 2).sum() / (n * (n ** 2 - 1))
    
    # Mean relative error
    mean_rel_error = ((D_true - D_approx).abs() / (D_true.abs() + 1e-10)).mean().item()
    
    return {
        'mse': mse,
        'relative_frobenius_error': rel_frob,
        'max_absolute_error': max_abs,
        'spearman_correlation': spearman,
        'mean_relative_error': mean_rel_error,
    }


def run_single_experiment(
    n: int,
    d: int,
    rank: int,
    epsilon: float,
    distribution: str,
    device: str,
) -> Dict[str, Dict[str, float]]:
    """Run a single experiment comparing estimators."""
    # Generate data
    X = generate_synthetic_data(n, d, distribution, device)
    
    # Compute true squared distances
    D_true = compute_squared_euclidean_distance_matrix(X)
    
    results = {}
    
    # Move to CPU for compatibility (MPS has some unsupported ops)
    X_cpu = X.cpu()
    D_true_cpu = D_true.cpu()
    
    # Low-rank approximation (main method from paper)
    low_rank = LowRankDistanceApproximator(rank=rank, epsilon=epsilon)
    D_low_rank = low_rank(X_cpu)
    # The low-rank method returns distances, not squared distances
    # Square it for comparison
    D_low_rank_sq = D_low_rank ** 2
    results['low_rank'] = compute_metrics(D_true_cpu, D_low_rank_sq)
    
    # Johnson-Lindenstrauss
    target_dim = min(rank, d)
    jl = JLDistanceEstimator(input_dim=d, target_dim=target_dim, epsilon=epsilon)
    D_jl = jl(X_cpu)
    results['jl'] = compute_metrics(D_true_cpu, D_jl)
    
    # Nyström (kernel approximation)
    nystrom = NystromDistanceEstimator(num_landmarks=rank)
    # Nyström returns kernel, convert to distances
    # K = exp(-D²/2σ²), so D² = -2σ² log(K)
    K_nystrom = nystrom(X_cpu)
    # For now, just compare kernel quality instead of distance
    # True kernel
    sigma = 1.0
    K_true = torch.exp(-D_true_cpu / (2 * sigma ** 2))
    results['nystrom_kernel'] = compute_metrics(K_true, K_nystrom)
    
    return results


def run_scaling_experiment(config: ExperimentConfig) -> Dict:
    """Run experiment varying sequence length."""
    print("=" * 60)
    print("Experiment: Distance Approximation Quality vs Sequence Length")
    print("=" * 60)
    
    torch.manual_seed(config.seed)
    
    results = {n: [] for n in config.seq_lengths}
    
    fixed_d = 64
    fixed_k = 32
    
    for n in tqdm(config.seq_lengths, desc="Sequence lengths"):
        for trial in range(config.num_trials):
            trial_results = run_single_experiment(
                n=n,
                d=fixed_d,
                rank=fixed_k,
                epsilon=config.epsilon,
                distribution='transformer_like',
                device=config.device,
            )
            results[n].append(trial_results)
    
    return results


def run_rank_experiment(config: ExperimentConfig) -> Dict:
    """Run experiment varying approximation rank."""
    print("=" * 60)
    print("Experiment: Distance Approximation Quality vs Rank")
    print("=" * 60)
    
    torch.manual_seed(config.seed)
    
    results = {k: [] for k in config.ranks}
    
    fixed_n = 256
    fixed_d = 64
    
    for k in tqdm(config.ranks, desc="Ranks"):
        for trial in range(config.num_trials):
            trial_results = run_single_experiment(
                n=fixed_n,
                d=fixed_d,
                rank=k,
                epsilon=config.epsilon,
                distribution='transformer_like',
                device=config.device,
            )
            results[k].append(trial_results)
    
    return results


def plot_results(results: Dict, x_label: str, title: str, save_path: Path):
    """Plot experiment results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    methods = ['low_rank', 'jl', 'nystrom_kernel']
    metrics = ['relative_frobenius_error', 'spearman_correlation', 'mse', 'mean_relative_error']
    metric_labels = ['Relative Frobenius Error', 'Spearman Correlation', 'MSE', 'Mean Relative Error']
    
    x_values = sorted(results.keys())
    
    colors = {'low_rank': '#2ecc71', 'jl': '#3498db', 'nystrom_kernel': '#e74c3c'}
    
    for ax, metric, label in zip(axes.flatten(), metrics, metric_labels):
        for method in methods:
            means = []
            stds = []
            for x in x_values:
                values = [r[method][metric] for r in results[x]]
                means.append(np.mean(values))
                stds.append(np.std(values))
            
            means = np.array(means)
            stds = np.array(stds)
            
            ax.plot(x_values, means, 'o-', label=method, color=colors[method], linewidth=2)
            ax.fill_between(x_values, means - stds, means + stds, alpha=0.2, color=colors[method])
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if metric != 'spearman_correlation':
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def print_summary(results: Dict, param_name: str):
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"Summary: Varying {param_name}")
    print('=' * 60)
    
    x_values = sorted(results.keys())
    
    for method in ['low_rank', 'jl', 'nystrom_kernel']:
        print(f"\n{method.upper()}:")
        print(f"{'Param':<10} {'Rel.Frob':<12} {'Spearman':<12} {'MSE':<12}")
        print("-" * 50)
        
        for x in x_values:
            rel_frob = np.mean([r[method]['relative_frobenius_error'] for r in results[x]])
            spearman = np.mean([r[method]['spearman_correlation'] for r in results[x]])
            mse = np.mean([r[method]['mse'] for r in results[x]])
            print(f"{x:<10} {rel_frob:<12.4f} {spearman:<12.4f} {mse:<12.4f}")


def main():
    """Run all distance approximation experiments."""
    print("\n" + "=" * 60)
    print("DISTANCE APPROXIMATION QUALITY EXPERIMENTS")
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
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'distance_approximation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = ExperimentConfig(
        seq_lengths=[64, 128, 256, 512, 1024],
        ranks=[8, 16, 32, 64, 128],
        num_trials=3,
        device=device,
    )
    
    # Experiment 1: Varying sequence length
    print("\n[1/2] Running sequence length experiment...")
    scaling_results = run_scaling_experiment(config)
    print_summary(scaling_results, "sequence length")
    plot_results(
        scaling_results,
        x_label='Sequence Length (n)',
        title='Distance Approximation Quality vs Sequence Length',
        save_path=results_dir / 'scaling_experiment.png'
    )
    
    # Experiment 2: Varying rank
    print("\n[2/2] Running rank experiment...")
    rank_results = run_rank_experiment(config)
    print_summary(rank_results, "approximation rank")
    plot_results(
        rank_results,
        x_label='Approximation Rank (k)',
        title='Distance Approximation Quality vs Approximation Rank',
        save_path=results_dir / 'rank_experiment.png'
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

