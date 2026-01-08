"""
Approximate Euclidean Attention with Linear-Time Distance Estimation

This is the core contribution: using the low-rank distance matrix approximation
from Indyk et al. to achieve linear-time attention.

Key idea:
1. Sample k landmark tokens from the sequence
2. Compute distances from all tokens to landmarks: O(n*k)
3. Approximate full distance matrix via Nyström: D ≈ C @ W^{+} @ C.T
4. Apply softmax and compute attention output

Complexity: O(n*k*d + k³) instead of O(n²*d)
For k = O(√n) or k = O(log n), this is subquadratic!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math


class ApproximateEuclideanAttention(nn.Module):
    """
    Euclidean attention with linear-time distance approximation.
    
    Uses Nyström approximation to avoid computing all n² distances.
    
    Theory (from Indyk et al.):
    - Sample O(k/ε) landmarks
    - Read O(n*k/ε) distance entries  
    - Achieve: ||D - D_approx||²_F ≤ ||D - D_k||²_F + ε||D||²_F
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        num_landmarks: int = 64,
        landmark_method: Literal['uniform', 'learned', 'kmeans'] = 'uniform',
        temperature: Optional[float] = None,
        learnable_temperature: bool = True,
        regularization: float = 1e-6,
    ):
        """
        Args:
            embed_dim: Model dimension
            num_heads: Number of attention heads
            dropout: Attention dropout
            bias: Use bias in projections
            num_landmarks: Number of landmark points k
            landmark_method: How to select landmarks
            temperature: Fixed temperature (if None, learned)
            learnable_temperature: Whether to learn temperature
            regularization: Regularization for matrix inversion
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.num_landmarks = num_landmarks
        self.landmark_method = landmark_method
        self.regularization = regularization
        
        assert embed_dim % num_heads == 0
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Temperature
        if temperature is not None:
            self.register_buffer('temperature', torch.tensor(temperature))
        elif learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(float(self.head_dim)))
        else:
            self.register_buffer('temperature', torch.tensor(float(self.head_dim)))
            
        # Learned landmarks (optional)
        if landmark_method == 'learned':
            self.landmark_queries = nn.Parameter(
                torch.randn(num_heads, num_landmarks, self.head_dim) * 0.02
            )
    
    def _select_landmark_indices(
        self, 
        seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Select landmark indices uniformly."""
        k = min(self.num_landmarks, seq_len)
        return torch.randperm(seq_len, device=device)[:k]
    
    def _compute_distances_to_landmarks(
        self,
        X: torch.Tensor,
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute squared distances from all tokens to landmark tokens.
        
        Args:
            X: (batch, heads, seq_len, head_dim)
            landmarks: (batch, heads, k, head_dim)
            
        Returns:
            C: (batch, heads, seq_len, k) distances to landmarks
        """
        # ||x||²: (batch, heads, seq_len, 1)
        X_sqnorm = (X ** 2).sum(dim=-1, keepdim=True)
        
        # ||l||²: (batch, heads, 1, k)
        L_sqnorm = (landmarks ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        
        # ⟨x, l⟩: (batch, heads, seq_len, k)
        XL = torch.matmul(X, landmarks.transpose(-2, -1))
        
        # D² = ||x||² + ||l||² - 2⟨x, l⟩
        D_sq = X_sqnorm + L_sqnorm - 2 * XL
        D_sq = torch.clamp(D_sq, min=0.0)
        
        return D_sq
    
    def _nystrom_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        landmark_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention using Nyström approximation.
        
        Instead of computing n×n attention matrix:
        1. Compute C_Q: distances from Q to landmarks (n, k)
        2. Compute C_K: distances from K to landmarks (n, k)
        3. Compute W: distances among landmarks (k, k)
        4. Attention ≈ softmax(-C_Q @ W^{+} @ C_K.T / τ)
        
        But this is still O(n²) due to the final matmul!
        
        Better approach: Use kernel formulation
        K(q,k) = exp(-||q-k||²/τ)
        ≈ exp(-||q-l||²/τ) @ W^{-1} @ exp(-||l-k||²/τ)
        
        This gives: A ≈ Φ_Q @ Φ_K.T where Φ = exp(-C²/τ) @ W^{-1/2}
        Output = softmax(A) @ V ≈ normalize(Φ_Q @ (Φ_K.T @ V))
        
        This is O(n*k)!
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        k = landmark_indices.shape[0]
        
        # Get landmark vectors
        # landmark_indices: (k,)
        # K is (batch, heads, seq_len, head_dim)
        landmarks = K[:, :, landmark_indices, :]  # (batch, heads, k, head_dim)
        
        # Compute distances to landmarks
        # C_Q: (batch, heads, seq_len, k) - Q to landmarks
        C_Q_sq = self._compute_distances_to_landmarks(Q, landmarks)
        
        # C_K: (batch, heads, seq_len, k) - K to landmarks  
        C_K_sq = self._compute_distances_to_landmarks(K, landmarks)
        
        # W: (batch, heads, k, k) - landmark to landmark
        W_sq = C_K_sq[:, :, landmark_indices, :]  # (batch, heads, k, k)
        
        # Convert to kernels: K(x,y) = exp(-D²/τ)
        Phi_Q = torch.exp(-C_Q_sq / self.temperature)  # (batch, heads, seq_len, k)
        Phi_K = torch.exp(-C_K_sq / self.temperature)  # (batch, heads, seq_len, k)
        W_kernel = torch.exp(-W_sq / self.temperature)  # (batch, heads, k, k)
        
        # Regularize W
        eye = torch.eye(k, device=W_kernel.device, dtype=W_kernel.dtype)
        W_kernel = W_kernel + self.regularization * eye
        
        # Compute W^{-1}
        # Using Cholesky for positive definite matrices
        try:
            L = torch.linalg.cholesky(W_kernel)  # (batch, heads, k, k)
            W_inv = torch.cholesky_inverse(L)
        except:
            # Fallback to pseudoinverse
            W_inv = torch.linalg.pinv(W_kernel)
        
        # Linear attention trick:
        # A ≈ Phi_Q @ W^{-1} @ Phi_K.T  (n×k × k×k × k×n = n×n, still quadratic!)
        #
        # But for output: out = A @ V = Phi_Q @ W^{-1} @ (Phi_K.T @ V)
        # Let Z = Phi_K.T @ V  (k×n × n×d = k×d)
        # Let Y = W^{-1} @ Z   (k×k × k×d = k×d)
        # out = Phi_Q @ Y      (n×k × k×d = n×d)
        #
        # This is O(n*k*d) instead of O(n²)!
        
        # Z = Phi_K.T @ V: (batch, heads, k, head_dim)
        Z = torch.matmul(Phi_K.transpose(-2, -1), V)
        
        # Y = W^{-1} @ Z: (batch, heads, k, head_dim)
        Y = torch.matmul(W_inv, Z)
        
        # Unnormalized output: (batch, heads, seq_len, head_dim)
        output_unnorm = torch.matmul(Phi_Q, Y)
        
        # Normalization (approximate softmax denominator)
        # Sum of attention weights per row ≈ Phi_Q @ W^{-1} @ Phi_K.T @ 1
        # = Phi_Q @ W^{-1} @ (sum of Phi_K rows)
        Phi_K_sum = Phi_K.sum(dim=-2, keepdim=True).transpose(-2, -1)  # (batch, heads, k, 1)
        norm_factor = torch.matmul(Phi_Q, torch.matmul(W_inv, Phi_K_sum))  # (batch, heads, seq_len, 1)
        norm_factor = torch.clamp(norm_factor, min=1e-10)
        
        # Normalize
        output = output_unnorm / norm_factor
        
        return output, None  # Don't return attention weights (would require O(n²))
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with approximate Euclidean attention.
        
        Note: attn_mask is not fully supported in the approximation.
        For causal masking, consider using separate landmarks per position.
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size, seq_len, _ = query.shape
        
        # Project
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Select landmarks
        if self.landmark_method == 'learned':
            # Use learned landmark queries
            # Not implemented in this version
            landmark_indices = self._select_landmark_indices(seq_len, Q.device)
        else:
            landmark_indices = self._select_landmark_indices(seq_len, Q.device)
        
        # Compute approximate attention
        output, attn_weights = self._nystrom_attention(Q, K, V, landmark_indices)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def compute_complexity(self, seq_len: int) -> dict:
        """
        Compute complexity for this attention mechanism.
        """
        k = min(self.num_landmarks, seq_len)
        d = self.head_dim
        h = self.num_heads
        n = seq_len
        
        # Distance to landmarks: O(n * k * d)
        dist_flops = 2 * n * k * d * h
        
        # Kernel matrix W inverse: O(k³)
        inv_flops = k ** 3 * h
        
        # Z = Phi_K.T @ V: O(k * n * d)
        z_flops = 2 * k * n * d * h
        
        # Y = W^{-1} @ Z: O(k² * d)
        y_flops = 2 * k * k * d * h
        
        # Output = Phi_Q @ Y: O(n * k * d)
        out_flops = 2 * n * k * d * h
        
        return {
            'distance_to_landmarks': dist_flops,
            'kernel_inverse': inv_flops,
            'kv_computation': z_flops,
            'intermediate': y_flops,
            'output': out_flops,
            'total': dist_flops + inv_flops + z_flops + y_flops + out_flops,
            'complexity': f'O(n*k*d + k³) = O({n}*{k}*{d} + {k}³)',
        }

