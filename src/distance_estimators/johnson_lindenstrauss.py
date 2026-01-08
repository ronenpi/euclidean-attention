"""
Johnson-Lindenstrauss Distance Estimation

The JL Lemma: For any ε ∈ (0,1) and any set of n points in ℝ^d,
there exists a map f: ℝ^d → ℝ^k where k = O(log(n)/ε²) such that
for all pairs x, y:

    (1-ε)||x-y||² ≤ ||f(x)-f(y)||² ≤ (1+ε)||x-y||²

Key insight: Random projections preserve distances!
This allows us to project Q, K to lower dimension, then compute
distances in the reduced space.
"""
import torch
import torch.nn as nn
from typing import Optional, Literal
import math


class JLDistanceEstimator(nn.Module):
    """
    Johnson-Lindenstrauss random projection for distance estimation.
    
    Projects d-dimensional vectors to k-dimensional space where
    k = O(log(n)/ε²), then computes exact distances in reduced space.
    
    Note: This still requires O(n²) distance computations, but each
    computation is in dimension k << d. The main benefit is:
    1. Memory: Store k-dim projections instead of d-dim
    2. Speed: Distance computation is O(k) instead of O(d)
    
    For truly sublinear algorithms, combine with low-rank approximation.
    """
    
    def __init__(
        self,
        input_dim: int,
        target_dim: Optional[int] = None,
        epsilon: float = 0.1,
        projection_type: Literal['gaussian', 'sparse', 'srht'] = 'gaussian',
        seed: Optional[int] = None,
    ):
        """
        Args:
            input_dim: Original dimension d
            target_dim: Projected dimension k. If None, computed from epsilon.
            epsilon: Distortion parameter (smaller = more accurate, larger k)
            projection_type: Type of random projection
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.projection_type = projection_type
        
        # k = O(log(n)/ε²) - but n is unknown at init time
        # Use a reasonable default or specified target_dim
        if target_dim is None:
            # Assume n ≈ 1000, gives k ≈ 4*ln(1000)/0.1² ≈ 2764
            # In practice, much smaller k often works
            target_dim = max(16, int(4 * math.log(1000) / (epsilon ** 2)))
            target_dim = min(target_dim, input_dim)  # Don't exceed input dim
        
        self.target_dim = target_dim
        
        # Initialize projection matrix
        if seed is not None:
            torch.manual_seed(seed)
            
        self.register_buffer(
            'projection_matrix',
            self._init_projection_matrix()
        )
        
    def _init_projection_matrix(self) -> torch.Tensor:
        """Initialize the random projection matrix."""
        d, k = self.input_dim, self.target_dim
        
        if self.projection_type == 'gaussian':
            # Standard Gaussian: P[i,j] ~ N(0, 1/k)
            P = torch.randn(d, k) / math.sqrt(k)
            
        elif self.projection_type == 'sparse':
            # Sparse random projection (Achlioptas)
            # P[i,j] = sqrt(3)/sqrt(k) * {+1 w.p. 1/6, 0 w.p. 2/3, -1 w.p. 1/6}
            probs = torch.rand(d, k)
            P = torch.zeros(d, k)
            P[probs < 1/6] = 1.0
            P[probs > 5/6] = -1.0
            P = P * math.sqrt(3.0 / k)
            
        elif self.projection_type == 'srht':
            # Subsampled Randomized Hadamard Transform
            # Faster but requires d to be power of 2
            # For simplicity, fall back to Gaussian if d is not power of 2
            if d & (d - 1) == 0:  # d is power of 2
                P = self._srht_matrix(d, k)
            else:
                P = torch.randn(d, k) / math.sqrt(k)
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
            
        return P
    
    def _srht_matrix(self, d: int, k: int) -> torch.Tensor:
        """
        Subsampled Randomized Hadamard Transform.
        
        SRHT(x) = sqrt(d/k) * P @ H @ D @ x
        
        where:
        - D: diagonal matrix with random ±1
        - H: Hadamard matrix
        - P: random sampling matrix (selects k rows)
        """
        # Random signs
        signs = 2 * torch.randint(0, 2, (d,)).float() - 1  # ±1
        
        # Hadamard matrix (recursive construction)
        H = self._hadamard(d)
        
        # Random row selection
        perm = torch.randperm(d)[:k]
        
        # Combine: scale * H @ diag(signs), then select rows
        scaled_H = math.sqrt(d / k) * H
        signed_H = scaled_H * signs.unsqueeze(0)  # Broadcast signs across rows
        
        return signed_H[perm].T  # (d, k)
    
    def _hadamard(self, n: int) -> torch.Tensor:
        """Construct n×n Hadamard matrix (n must be power of 2)."""
        if n == 1:
            return torch.tensor([[1.0]])
        else:
            H_half = self._hadamard(n // 2)
            H = torch.cat([
                torch.cat([H_half, H_half], dim=1),
                torch.cat([H_half, -H_half], dim=1)
            ], dim=0) / math.sqrt(2)
            return H
    
    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project vectors to lower-dimensional space.
        
        Args:
            X: (..., d) input vectors
            
        Returns:
            X_proj: (..., k) projected vectors
        """
        # Move projection matrix to same device as input
        proj = self.projection_matrix.to(X.device)
        return X @ proj
    
    def estimate_distances(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        squared: bool = True
    ) -> torch.Tensor:
        """
        Estimate pairwise distances using random projection.
        
        Args:
            X: (n, d) query vectors
            Y: (m, d) key vectors (if None, Y = X)
            squared: if True, return squared distances
            
        Returns:
            D: (n, m) estimated distance matrix
        """
        # Project to lower dimension
        X_proj = self.project(X)  # (n, k)
        
        if Y is None:
            Y_proj = X_proj
        else:
            Y_proj = self.project(Y)  # (m, k)
        
        # Compute distances in projected space
        X_sqnorm = (X_proj ** 2).sum(dim=-1, keepdim=True)  # (n, 1)
        Y_sqnorm = (Y_proj ** 2).sum(dim=-1, keepdim=True).T  # (1, m)
        XY = X_proj @ Y_proj.T  # (n, m)
        
        D_sq = X_sqnorm + Y_sqnorm - 2 * XY
        D_sq = torch.clamp(D_sq, min=0.0)
        
        if squared:
            return D_sq
        else:
            return torch.sqrt(D_sq)
    
    def forward(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass: return squared distance estimates."""
        return self.estimate_distances(X, Y, squared=True)


def jl_dimension(n: int, epsilon: float, delta: float = 0.01) -> int:
    """
    Compute target dimension for JL projection.
    
    k ≥ 4 * ln(n/δ) / (ε² - ε³/3)
    
    Args:
        n: Number of points
        epsilon: Distortion parameter
        delta: Failure probability
        
    Returns:
        k: Target dimension
    """
    return int(math.ceil(
        4 * math.log(n / delta) / (epsilon ** 2 - epsilon ** 3 / 3)
    ))

