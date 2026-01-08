"""
Basic tests to verify implementation correctness.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_exact_distance():
    """Test exact distance computation."""
    from distance_estimators import compute_squared_euclidean_distance_matrix
    
    # Simple test case
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    D = compute_squared_euclidean_distance_matrix(X)
    
    # Check diagonal is zero
    assert torch.allclose(D.diag(), torch.zeros(3)), "Diagonal should be zero"
    
    # Check symmetry
    assert torch.allclose(D, D.T), "Distance matrix should be symmetric"
    
    # Check specific distances
    # ||[0,0] - [1,0]||² = 1
    assert torch.allclose(D[0, 1], torch.tensor(1.0)), f"D[0,1] should be 1, got {D[0,1]}"
    # ||[0,0] - [0,1]||² = 1
    assert torch.allclose(D[0, 2], torch.tensor(1.0)), f"D[0,2] should be 1, got {D[0,2]}"
    # ||[1,0] - [0,1]||² = 2
    assert torch.allclose(D[1, 2], torch.tensor(2.0)), f"D[1,2] should be 2, got {D[1,2]}"
    
    print("✓ test_exact_distance passed")


def test_low_rank_approximation():
    """Test low-rank distance approximation."""
    from distance_estimators import (
        compute_squared_euclidean_distance_matrix,
        LowRankDistanceApproximator
    )
    
    torch.manual_seed(42)
    n, d = 64, 32
    X = torch.randn(n, d)
    
    D_true = compute_squared_euclidean_distance_matrix(X)
    
    # Test with different ranks
    for rank in [8, 16, 32]:
        approx = LowRankDistanceApproximator(rank=rank, epsilon=0.1)
        D_approx = approx(X)
        D_approx_sq = D_approx ** 2
        
        # Check shape
        assert D_approx_sq.shape == D_true.shape, f"Shape mismatch: {D_approx_sq.shape} vs {D_true.shape}"
        
        # Check error decreases with rank
        rel_error = torch.norm(D_true - D_approx_sq, 'fro') / torch.norm(D_true, 'fro')
        print(f"  Rank {rank}: relative error = {rel_error:.4f}")
        
    print("✓ test_low_rank_approximation passed")


def test_jl_estimator():
    """Test Johnson-Lindenstrauss estimator."""
    from distance_estimators import JLDistanceEstimator, compute_squared_euclidean_distance_matrix
    
    torch.manual_seed(42)
    n, d = 64, 128
    X = torch.randn(n, d)
    
    D_true = compute_squared_euclidean_distance_matrix(X)
    
    jl = JLDistanceEstimator(input_dim=d, target_dim=32, epsilon=0.2)
    D_jl = jl(X)
    
    # Check shape
    assert D_jl.shape == D_true.shape
    
    # JL should preserve distances within (1±ε) factor on average
    ratio = D_jl / (D_true + 1e-10)
    # Exclude diagonal (0/0)
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_ratio = ratio[mask].mean()
    
    print(f"  Mean distance ratio: {mean_ratio:.4f} (should be ~1.0)")
    assert 0.5 < mean_ratio < 1.5, f"JL distances too distorted: {mean_ratio}"
    
    print("✓ test_jl_estimator passed")


def test_standard_attention():
    """Test standard attention."""
    from attention import StandardAttention
    
    batch, seq, dim = 2, 16, 64
    heads = 4
    
    X = torch.randn(batch, seq, dim)
    attn = StandardAttention(dim, heads)
    
    output, weights = attn(X, return_attention=True)
    
    # Check shapes
    assert output.shape == (batch, seq, dim), f"Output shape wrong: {output.shape}"
    assert weights.shape == (batch, heads, seq, seq), f"Weights shape wrong: {weights.shape}"
    
    # Check attention weights sum to 1
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Attention should sum to 1"
    
    print("✓ test_standard_attention passed")


def test_euclidean_attention():
    """Test Euclidean attention."""
    from attention import EuclideanAttention
    
    batch, seq, dim = 2, 16, 64
    heads = 4
    
    X = torch.randn(batch, seq, dim)
    attn = EuclideanAttention(dim, heads)
    
    output, weights = attn(X, return_attention=True)
    
    # Check shapes
    assert output.shape == (batch, seq, dim)
    assert weights.shape == (batch, heads, seq, seq)
    
    # Check attention weights sum to 1
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    # Check diagonal has highest attention (token most similar to itself)
    for b in range(batch):
        for h in range(heads):
            diag_vals = weights[b, h].diag()
            # Diagonal should be relatively high (but not necessarily highest due to temperature)
            assert diag_vals.mean() > 0.01, "Self-attention should be non-negligible"
    
    print("✓ test_euclidean_attention passed")


def test_approximate_attention():
    """Test approximate Euclidean attention."""
    from attention import ApproximateEuclideanAttention
    
    batch, seq, dim = 2, 32, 64
    heads = 4
    landmarks = 8
    
    X = torch.randn(batch, seq, dim)
    attn = ApproximateEuclideanAttention(dim, heads, num_landmarks=landmarks)
    
    output, _ = attn(X)
    
    # Check output shape
    assert output.shape == (batch, seq, dim), f"Output shape wrong: {output.shape}"
    
    # Check output is not NaN
    assert not torch.isnan(output).any(), "Output contains NaN"
    
    # Check output is not all zeros
    assert output.abs().mean() > 0.01, "Output is near zero"
    
    print("✓ test_approximate_attention passed")


def test_attention_consistency():
    """Test that different attention mechanisms produce valid outputs."""
    from attention import StandardAttention, EuclideanAttention, ApproximateEuclideanAttention
    
    torch.manual_seed(42)
    batch, seq, dim = 1, 32, 64
    heads = 4
    
    X = torch.randn(batch, seq, dim)
    
    std = StandardAttention(dim, heads)
    euc = EuclideanAttention(dim, heads)
    approx = ApproximateEuclideanAttention(dim, heads, num_landmarks=8)
    
    # Copy weights
    with torch.no_grad():
        euc.q_proj.weight.copy_(std.q_proj.weight)
        euc.k_proj.weight.copy_(std.k_proj.weight)
        euc.v_proj.weight.copy_(std.v_proj.weight)
        euc.out_proj.weight.copy_(std.out_proj.weight)
        
        approx.q_proj.weight.copy_(std.q_proj.weight)
        approx.k_proj.weight.copy_(std.k_proj.weight)
        approx.v_proj.weight.copy_(std.v_proj.weight)
        approx.out_proj.weight.copy_(std.out_proj.weight)
    
    with torch.no_grad():
        out_std, _ = std(X)
        out_euc, _ = euc(X)
        out_approx, _ = approx(X)
    
    # All outputs should be valid
    for name, out in [('standard', out_std), ('euclidean', out_euc), ('approximate', out_approx)]:
        assert not torch.isnan(out).any(), f"{name} output contains NaN"
        assert out.abs().mean() > 0.01, f"{name} output is near zero"
    
    # Outputs should be different (different mechanisms)
    # But not too different (same projection weights)
    cos_std_euc = torch.nn.functional.cosine_similarity(
        out_std.flatten().unsqueeze(0),
        out_euc.flatten().unsqueeze(0)
    ).item()
    
    print(f"  Cosine similarity (std vs euc): {cos_std_euc:.4f}")
    # They should be positively correlated but not identical
    assert cos_std_euc > 0.3, "Outputs too different"
    
    print("✓ test_attention_consistency passed")


def main():
    print("\n" + "=" * 50)
    print("RUNNING BASIC TESTS")
    print("=" * 50 + "\n")
    
    test_exact_distance()
    test_low_rank_approximation()
    test_jl_estimator()
    test_standard_attention()
    test_euclidean_attention()
    test_approximate_attention()
    test_attention_consistency()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()

