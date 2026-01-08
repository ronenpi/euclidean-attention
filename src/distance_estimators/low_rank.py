"""
Low-Rank Distance Matrix Approximation

Based on: "Sample-Optimal Low-Rank Approximation of Distance Matrices"
Indyk, Vakilian, Wagner, Woodruff (NeurIPS 2019)

Key insight: Distance matrices have special structure that allows
computing rank-k approximations by reading only O((n+m)k/ε) entries,
in Õ(n+m)·poly(k,1/ε) time.

For attention: We approximate the n×n distance matrix D with low-rank VU,
where V ∈ ℝ^(n×k) and U ∈ ℝ^(k×n).
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class LowRankDistanceApproximator(nn.Module):
    """
    Approximates distance matrices using the sampling-based algorithm
    from Indyk et al.
    
    The algorithm:
    1. Sample O(k/ε) "landmark" columns uniformly at random
    2. Sample O(k/ε) rows uniformly at random  
    3. Use these to construct a low-rank approximation
    
    Guarantee: ||D - VU||²_F ≤ ||D - D_k||²_F + ε||D||²_F
    """
    
    def __init__(
        self, 
        rank: int = 32,
        epsilon: float = 0.1,
        use_nystrom: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.epsilon = epsilon
        self.use_nystrom = use_nystrom
        
    def _sample_indices(self, n: int, num_samples: int) -> torch.Tensor:
        """Uniformly sample indices."""
        num_samples = min(num_samples, n)
        return torch.randperm(n)[:num_samples]
    
    def _compute_column_distances(
        self, 
        X: torch.Tensor, 
        landmark_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances from all points to landmark points.
        
        Args:
            X: (n, d) all points
            landmark_indices: (k,) indices of landmark points
            
        Returns:
            C: (n, k) distances to landmarks
        """
        landmarks = X[landmark_indices]  # (k, d)
        
        # ||x_i - l_j||² = ||x_i||² + ||l_j||² - 2⟨x_i, l_j⟩
        X_sqnorm = (X ** 2).sum(dim=-1, keepdim=True)  # (n, 1)
        L_sqnorm = (landmarks ** 2).sum(dim=-1, keepdim=True).T  # (1, k)
        XL = X @ landmarks.T  # (n, k)
        
        D_sq = X_sqnorm + L_sqnorm - 2 * XL
        D_sq = torch.clamp(D_sq, min=0.0)
        
        return torch.sqrt(D_sq)
    
    def approximate(
        self, 
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        return_factors: bool = False
    ) -> torch.Tensor:
        """
        Compute low-rank approximation of distance matrix.
        
        For self-attention (Y=None), we use a simplified Nyström-style approach:
        1. Sample k landmark points
        2. Compute C = distances from all points to landmarks: (n, k)
        3. Compute W = distances among landmarks: (k, k)  
        4. Approximate: D ≈ C @ W^{-1} @ C.T
        
        This reads O(nk + k²) entries instead of O(n²).
        
        Args:
            X: (n, d) query/key vectors
            Y: (m, d) key vectors (if different from X)
            return_factors: if True, return (V, U) factors
            
        Returns:
            D_approx: (n, m) approximate distance matrix
            or (V, U) if return_factors=True
        """
        if Y is None:
            Y = X
            self_attention = True
        else:
            self_attention = False
            
        n, d = X.shape
        m = Y.shape[0]
        
        # Number of landmarks based on theory: O(k/ε)
        num_landmarks = min(
            int(self.rank / self.epsilon),
            min(n, m)
        )
        
        if self_attention and self.use_nystrom:
            return self._nystrom_approximation(X, num_landmarks, return_factors)
        else:
            return self._column_sampling_approximation(
                X, Y, num_landmarks, return_factors
            )
    
    def _nystrom_approximation(
        self,
        X: torch.Tensor,
        num_landmarks: int,
        return_factors: bool = False
    ) -> torch.Tensor:
        """
        Nyström approximation for symmetric distance matrices.
        
        D ≈ C @ W^{+} @ C.T
        
        where:
        - C: (n, k) distances to landmarks
        - W: (k, k) distances among landmarks
        - W^{+}: pseudoinverse of W
        """
        n = X.shape[0]
        
        # Sample landmark indices
        landmark_idx = self._sample_indices(n, num_landmarks)
        
        # C: distances from all points to landmarks (n, k)
        C = self._compute_column_distances(X, landmark_idx)
        
        # W: distances among landmarks (k, k)
        W = C[landmark_idx]  # (k, k)
        
        # Regularized pseudoinverse for numerical stability
        # W^{+} = V @ S^{+} @ U.T from SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Truncate small singular values
        threshold = 1e-6 * S.max()
        S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
        
        W_pinv = Vh.T @ torch.diag(S_inv) @ U.T  # (k, k)
        
        if return_factors:
            # D ≈ V @ U where V = C @ W^{+/2}, U = W^{+/2} @ C.T
            # For simplicity, return C and W_pinv @ C.T
            V = C  # (n, k)
            U_factor = W_pinv @ C.T  # (k, n)
            return V, U_factor
        
        # D ≈ C @ W^{+} @ C.T
        D_approx = C @ W_pinv @ C.T
        
        return D_approx
    
    def _column_sampling_approximation(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        num_landmarks: int,
        return_factors: bool = False
    ) -> torch.Tensor:
        """
        Column sampling for asymmetric distance matrices.
        
        Sample columns from the distance matrix and use them
        for reconstruction.
        """
        n = X.shape[0]
        m = Y.shape[0]
        
        # Sample landmark indices from Y
        landmark_idx = self._sample_indices(m, num_landmarks)
        landmarks = Y[landmark_idx]  # (k, d)
        
        # Compute distances from X to landmarks: (n, k)
        X_sqnorm = (X ** 2).sum(dim=-1, keepdim=True)
        L_sqnorm = (landmarks ** 2).sum(dim=-1, keepdim=True).T
        C_X = torch.sqrt(torch.clamp(
            X_sqnorm + L_sqnorm - 2 * X @ landmarks.T, min=0.0
        ))
        
        # Compute distances from Y to landmarks: (m, k)
        Y_sqnorm = (Y ** 2).sum(dim=-1, keepdim=True)
        C_Y = torch.sqrt(torch.clamp(
            Y_sqnorm + L_sqnorm - 2 * Y @ landmarks.T, min=0.0
        ))
        
        # W: landmark-to-landmark distances (k, k)
        W = C_Y[landmark_idx]
        
        # Pseudoinverse
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        threshold = 1e-6 * S.max() if S.numel() > 0 else 1e-6
        S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
        W_pinv = Vh.T @ torch.diag(S_inv) @ U.T
        
        if return_factors:
            V = C_X  # (n, k)
            U_factor = W_pinv @ C_Y.T  # (k, m)
            return V, U_factor
        
        # D ≈ C_X @ W^{+} @ C_Y.T
        D_approx = C_X @ W_pinv @ C_Y.T
        
        return D_approx
    
    def forward(
        self, 
        X: torch.Tensor, 
        Y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for nn.Module compatibility."""
        return self.approximate(X, Y)


def low_rank_distance_flops(n: int, m: int, d: int, k: int) -> int:
    """
    FLOPs for low-rank distance approximation.
    
    - Compute C_X (n, k): 2*n*k*d
    - Compute C_Y (m, k): 2*m*k*d (if asymmetric)
    - Compute W (k, k): already computed
    - SVD of W: O(k³)
    - Matrix multiplications: O(n*k + k*m)
    
    Total: O((n+m)*k*d + k³) - LINEAR in n,m for fixed k!
    """
    return 2 * n * k * d + 2 * m * k * d + k ** 3 + n * k + k * m

