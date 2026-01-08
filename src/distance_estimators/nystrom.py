"""
Nyström Approximation for Distance/Kernel Matrices

The Nyström method approximates a kernel matrix K using a subset of columns:
K ≈ C @ W^{+} @ C.T

where:
- C: (n, k) selected columns of K
- W: (k, k) intersection of selected rows and columns
- W^{+}: pseudoinverse of W

For distance matrices, we can use K = exp(-D²/σ²) (Gaussian kernel).
"""
import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple
import math


class NystromDistanceEstimator(nn.Module):
    """
    Nyström approximation for distance-based kernel matrices.
    
    Instead of approximating distances directly, approximates the
    kernel matrix K[i,j] = exp(-||x_i - x_j||² / σ²).
    
    This is useful because attention is softmax(-D²/τ), which is
    similar to a Gaussian kernel.
    """
    
    def __init__(
        self,
        num_landmarks: int = 64,
        sigma: float = 1.0,
        sampling_method: Literal['uniform', 'kmeans', 'leverage'] = 'uniform',
        regularization: float = 1e-6,
    ):
        """
        Args:
            num_landmarks: Number of landmark points (k)
            sigma: Kernel bandwidth parameter
            sampling_method: How to select landmarks
            regularization: Regularization for pseudoinverse
        """
        super().__init__()
        self.num_landmarks = num_landmarks
        self.sigma = sigma
        self.sampling_method = sampling_method
        self.regularization = regularization
        
    def _select_landmarks(
        self, 
        X: torch.Tensor, 
        num_landmarks: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select landmark points.
        
        Args:
            X: (n, d) input points
            num_landmarks: number of landmarks to select
            
        Returns:
            indices: (k,) landmark indices
            landmarks: (k, d) landmark points
        """
        n = X.shape[0]
        k = min(num_landmarks, n)
        
        if self.sampling_method == 'uniform':
            indices = torch.randperm(n, device=X.device)[:k]
            
        elif self.sampling_method == 'kmeans':
            # K-means++ initialization for better coverage
            indices = self._kmeans_plus_plus(X, k)
            
        elif self.sampling_method == 'leverage':
            # Leverage score sampling (requires SVD, more expensive)
            indices = self._leverage_sampling(X, k)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
            
        landmarks = X[indices]
        return indices, landmarks
    
    def _kmeans_plus_plus(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """K-means++ initialization for landmark selection."""
        n, d = X.shape
        device = X.device
        
        # First center: random
        indices = [torch.randint(n, (1,), device=device).item()]
        
        for _ in range(k - 1):
            # Compute distances to nearest center
            centers = X[indices]  # (len(indices), d)
            
            # Distance to each center
            X_sqnorm = (X ** 2).sum(dim=-1, keepdim=True)
            C_sqnorm = (centers ** 2).sum(dim=-1, keepdim=True).T
            D_sq = X_sqnorm + C_sqnorm - 2 * X @ centers.T  # (n, len(indices))
            D_sq = torch.clamp(D_sq, min=0.0)
            
            # Min distance to any center
            min_D_sq = D_sq.min(dim=1).values  # (n,)
            
            # Sample proportional to D²
            probs = min_D_sq / min_D_sq.sum()
            new_idx = torch.multinomial(probs, 1).item()
            indices.append(new_idx)
            
        return torch.tensor(indices, device=device)
    
    def _leverage_sampling(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """Leverage score sampling."""
        n = X.shape[0]
        
        # Compute leverage scores from SVD
        # This is expensive but gives better approximation quality
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        
        # Leverage score of row i is ||U[i, :]||²
        leverage_scores = (U[:, :min(k, U.shape[1])] ** 2).sum(dim=1)
        leverage_scores = leverage_scores / leverage_scores.sum()
        
        # Sample k indices
        indices = torch.multinomial(leverage_scores, k, replacement=False)
        return indices
    
    def _gaussian_kernel(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Compute Gaussian kernel matrix.
        
        K[i,j] = exp(-||x_i - y_j||² / (2σ²))
        """
        X_sqnorm = (X ** 2).sum(dim=-1, keepdim=True)
        Y_sqnorm = (Y ** 2).sum(dim=-1, keepdim=True).T
        D_sq = X_sqnorm + Y_sqnorm - 2 * X @ Y.T
        D_sq = torch.clamp(D_sq, min=0.0)
        
        K = torch.exp(-D_sq / (2 * sigma ** 2))
        return K
    
    def approximate_kernel(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate the Gaussian kernel matrix using Nyström.
        
        K ≈ C_X @ W^{+} @ C_Y.T
        
        Args:
            X: (n, d) query points
            Y: (m, d) key points (if None, Y = X)
            
        Returns:
            K_approx: (n, m) approximate kernel matrix
        """
        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False
            
        n = X.shape[0]
        m = Y.shape[0]
        k = min(self.num_landmarks, min(n, m))
        
        # Select landmarks from Y (or X if symmetric)
        landmark_indices, landmarks = self._select_landmarks(Y, k)
        
        # C_X: kernel from X to landmarks (n, k)
        C_X = self._gaussian_kernel(X, landmarks, self.sigma)
        
        # C_Y: kernel from Y to landmarks (m, k)
        if symmetric:
            C_Y = C_X
        else:
            C_Y = self._gaussian_kernel(Y, landmarks, self.sigma)
        
        # W: kernel among landmarks (k, k)
        W = C_Y[landmark_indices]
        
        # Regularized pseudoinverse
        W_reg = W + self.regularization * torch.eye(k, device=W.device)
        
        # Use Cholesky for symmetric positive definite W
        try:
            L = torch.linalg.cholesky(W_reg)
            # K ≈ C_X @ W^{-1} @ C_Y.T = C_X @ (L.T)^{-1} @ L^{-1} @ C_Y.T
            # Let Z = L^{-1} @ C_Y.T, then K ≈ C_X @ (L.T)^{-1} @ Z
            Z = torch.linalg.solve_triangular(L, C_Y.T, upper=False)
            K_approx = C_X @ torch.linalg.solve_triangular(L.T, Z, upper=True)
        except:
            # Fall back to SVD if Cholesky fails
            U, S, Vh = torch.linalg.svd(W_reg)
            S_inv = 1.0 / S
            W_inv = Vh.T @ torch.diag(S_inv) @ U.T
            K_approx = C_X @ W_inv @ C_Y.T
        
        return K_approx
    
    def approximate_attention_weights(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Approximate attention weights using Nyström kernel approximation.
        
        Standard attention: softmax(QK^T / √d)
        Euclidean attention: softmax(-||Q-K||² / τ)
        
        Note: softmax(-D²/τ) ∝ exp(-D²/τ) ∝ Gaussian kernel with σ² = τ/2
        
        Args:
            Q: (n, d) query vectors
            K: (m, d) key vectors
            temperature: temperature parameter τ
            
        Returns:
            A: (n, m) approximate attention weights (rows sum to 1)
        """
        # Set sigma to match temperature: σ² = τ/2
        sigma = math.sqrt(temperature / 2)
        
        # Store original sigma
        orig_sigma = self.sigma
        self.sigma = sigma
        
        # Approximate kernel (which is exp(-D²/2σ²) = exp(-D²/τ))
        K_approx = self.approximate_kernel(Q, K)
        
        # Restore original sigma
        self.sigma = orig_sigma
        
        # Normalize rows (softmax normalization)
        # Since exp(-D²/τ) is already non-negative, just normalize
        row_sums = K_approx.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-10)  # Avoid division by zero
        A = K_approx / row_sums
        
        return A
    
    def forward(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass: return approximate kernel matrix."""
        return self.approximate_kernel(X, Y)

