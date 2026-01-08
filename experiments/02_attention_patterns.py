"""
Experiment 2: Attention Pattern Analysis

Goal: Visualize and compare attention patterns between:
1. Standard dot-product attention
2. Exact Euclidean attention  
3. Approximate Euclidean attention

Questions to answer:
- Do Euclidean attention patterns look meaningfully different?
- Does the approximation preserve the attention structure?
- How does temperature affect Euclidean attention?
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from attention import StandardAttention, EuclideanAttention, ApproximateEuclideanAttention


@dataclass
class AttentionConfig:
    embed_dim: int = 64
    num_heads: int = 4
    seq_len: int = 64
    batch_size: int = 1
    num_landmarks: int = 16
    device: str = 'cpu'
    seed: int = 42


def generate_sequence_with_structure(config: AttentionConfig) -> torch.Tensor:
    """
    Generate a sequence with known structure for attention visualization.
    
    Creates a sequence where:
    - Some tokens are "anchors" that should attend broadly
    - Some tokens are "local" that should attend to neighbors
    - Some tokens are "copies" of earlier tokens
    """
    torch.manual_seed(config.seed)
    
    n = config.seq_len
    d = config.embed_dim
    
    # Base random embeddings
    X = torch.randn(config.batch_size, n, d, device=config.device)
    
    # Add positional structure (nearby tokens are more similar)
    positions = torch.arange(n, device=config.device).float()
    pos_encoding = torch.sin(positions.unsqueeze(-1) * torch.arange(d, device=config.device) * 0.1)
    X = X + 0.5 * pos_encoding.unsqueeze(0)
    
    # Create some "copy" tokens (exact duplicates should get high attention)
    copy_positions = [10, 30, 50]
    source_positions = [5, 25, 45]
    for copy_pos, src_pos in zip(copy_positions, source_positions):
        X[:, copy_pos] = X[:, src_pos].clone()
    
    return X


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention distribution (averaged over all positions).
    Higher entropy = more uniform attention.
    """
    # attn_weights: (batch, heads, seq, seq)
    # Add small epsilon for numerical stability
    attn_weights = attn_weights + 1e-10
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1)  # (batch, heads, seq)
    return entropy.mean().item()


def compute_attention_sparsity(attn_weights: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute sparsity of attention (fraction of weights below threshold).
    """
    sparse_count = (attn_weights < threshold).float().sum()
    total_count = attn_weights.numel()
    return (sparse_count / total_count).item()


def compute_locality_score(attn_weights: torch.Tensor) -> float:
    """
    Compute how "local" the attention is (higher = more local focus).
    
    Uses weighted average distance from diagonal.
    """
    # attn_weights: (batch, heads, seq, seq)
    n = attn_weights.shape[-1]
    
    # Distance from diagonal
    positions = torch.arange(n, device=attn_weights.device)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()  # (n, n)
    
    # Weighted average distance
    weighted_dist = (attn_weights * distances).sum(dim=-1)  # (batch, heads, seq)
    
    # Normalize by sequence length
    locality = 1.0 - weighted_dist.mean().item() / (n / 2)
    return locality


def analyze_attention(
    attn_weights: torch.Tensor,
    name: str
) -> Dict[str, float]:
    """Compute all attention metrics."""
    return {
        'name': name,
        'entropy': compute_attention_entropy(attn_weights),
        'sparsity': compute_attention_sparsity(attn_weights),
        'locality': compute_locality_score(attn_weights),
        'max_weight': attn_weights.max().item(),
        'min_weight': attn_weights.min().item(),
    }


def plot_attention_comparison(
    attention_dict: Dict[str, torch.Tensor],
    save_path: Path,
    head_idx: int = 0
):
    """Plot attention patterns side by side."""
    n_methods = len(attention_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (name, attn) in zip(axes, attention_dict.items()):
        # Take first batch, specified head
        attn_matrix = attn[0, head_idx].detach().cpu().numpy()
        
        sns.heatmap(
            attn_matrix,
            ax=ax,
            cmap='viridis',
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(f'{name}\nHead {head_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_statistics(
    stats_list: list,
    save_path: Path
):
    """Plot attention statistics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    names = [s['name'] for s in stats_list]
    metrics = ['entropy', 'sparsity', 'locality', 'max_weight']
    titles = ['Attention Entropy', 'Sparsity (< 0.01)', 'Locality Score', 'Max Attention Weight']
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'][:len(names)]
    
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        values = [s[metric] for s in stats_list]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_difference(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    name1: str,
    name2: str,
    save_path: Path,
    head_idx: int = 0
):
    """Plot the difference between two attention patterns."""
    diff = (attn1 - attn2)[0, head_idx].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Attention 1
    sns.heatmap(attn1[0, head_idx].detach().cpu().numpy(), ax=axes[0], 
                cmap='viridis', square=True)
    axes[0].set_title(name1)
    
    # Attention 2
    sns.heatmap(attn2[0, head_idx].detach().cpu().numpy(), ax=axes[1],
                cmap='viridis', square=True)
    axes[1].set_title(name2)
    
    # Difference
    vmax = max(abs(diff.min()), abs(diff.max()))
    sns.heatmap(diff, ax=axes[2], cmap='RdBu_r', center=0, 
                vmin=-vmax, vmax=vmax, square=True)
    axes[2].set_title(f'Difference ({name1} - {name2})')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def run_attention_comparison(config: AttentionConfig):
    """Compare attention mechanisms."""
    print("=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Generate input
    X = generate_sequence_with_structure(config)
    print(f"\nInput shape: {X.shape}")
    
    # Initialize attention mechanisms
    std_attn = StandardAttention(config.embed_dim, config.num_heads).to(config.device)
    euc_attn = EuclideanAttention(config.embed_dim, config.num_heads).to(config.device)
    approx_attn = ApproximateEuclideanAttention(
        config.embed_dim, config.num_heads,
        num_landmarks=config.num_landmarks
    ).to(config.device)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        euc_attn.q_proj.weight.copy_(std_attn.q_proj.weight)
        euc_attn.k_proj.weight.copy_(std_attn.k_proj.weight)
        euc_attn.v_proj.weight.copy_(std_attn.v_proj.weight)
        
        approx_attn.q_proj.weight.copy_(std_attn.q_proj.weight)
        approx_attn.k_proj.weight.copy_(std_attn.k_proj.weight)
        approx_attn.v_proj.weight.copy_(std_attn.v_proj.weight)
    
    # Compute attention
    with torch.no_grad():
        _, std_weights = std_attn(X, return_attention=True)
        _, euc_weights = euc_attn(X, return_attention=True)
        
        # Approximate attention doesn't return weights directly
        # Compute outputs and compare
        out_std, _ = std_attn(X)
        out_euc, _ = euc_attn(X)
        out_approx, _ = approx_attn(X)
    
    return {
        'standard': std_weights,
        'euclidean': euc_weights,
        'outputs': {
            'standard': out_std,
            'euclidean': out_euc,
            'approximate': out_approx,
        }
    }


def main():
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN ANALYSIS EXPERIMENTS")
    print("=" * 60 + "\n")
    
    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'attention_patterns'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = AttentionConfig(
        embed_dim=64,
        num_heads=4,
        seq_len=64,
        num_landmarks=16,
        device=device,
    )
    
    # Run comparison
    results = run_attention_comparison(config)
    
    # Analyze
    print("\n[1] Analyzing attention patterns...")
    
    stats = []
    for name in ['standard', 'euclidean']:
        if results[name] is not None:
            stat = analyze_attention(results[name], name)
            stats.append(stat)
            print(f"\n{name.upper()} Attention:")
            for k, v in stat.items():
                if k != 'name':
                    print(f"  {k}: {v:.4f}")
    
    # Plot
    print("\n[2] Generating visualizations...")
    
    attention_dict = {k: v for k, v in results.items() 
                      if k in ['standard', 'euclidean'] and v is not None}
    
    if len(attention_dict) >= 2:
        # Side by side comparison
        plot_attention_comparison(
            attention_dict,
            results_dir / 'attention_comparison.png'
        )
        
        # Statistics comparison
        plot_attention_statistics(stats, results_dir / 'attention_statistics.png')
        
        # Difference plot
        plot_attention_difference(
            results['standard'],
            results['euclidean'],
            'Standard',
            'Euclidean',
            results_dir / 'attention_difference.png'
        )
    
    # Compare outputs
    print("\n[3] Comparing output representations...")
    outputs = results['outputs']
    
    # Compute similarities between outputs
    out_std = outputs['standard'].flatten()
    out_euc = outputs['euclidean'].flatten()
    out_approx = outputs['approximate'].flatten()
    
    cos_std_euc = F.cosine_similarity(out_std.unsqueeze(0), out_euc.unsqueeze(0)).item()
    cos_std_approx = F.cosine_similarity(out_std.unsqueeze(0), out_approx.unsqueeze(0)).item()
    cos_euc_approx = F.cosine_similarity(out_euc.unsqueeze(0), out_approx.unsqueeze(0)).item()
    
    print(f"\nOutput Cosine Similarities:")
    print(f"  Standard vs Euclidean: {cos_std_euc:.4f}")
    print(f"  Standard vs Approximate: {cos_std_approx:.4f}")
    print(f"  Euclidean vs Approximate: {cos_euc_approx:.4f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

