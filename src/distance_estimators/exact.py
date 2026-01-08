"""
Exact Euclidean distance computation (baseline).
"""
import torch
from typing import Optional


def compute_squared_euclidean_distance_matrix(
    X: torch.Tensor, 
    Y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute squared Euclidean distance matrix between X and Y.
    
    D²[i,j] = ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2⟨x_i, y_j⟩
    
    Args:
        X: (n, d) tensor of query vectors
        Y: (m, d) tensor of key vectors. If None, computes self-distances.
        
    Returns:
        D²: (n, m) squared distance matrix
    """
    if Y is None:
        Y = X
        
    # ||x||² for each row of X: (n, 1)
    X_sqnorm = (X ** 2).sum(dim=-1, keepdim=True)
    
    # ||y||² for each row of Y: (1, m)
    Y_sqnorm = (Y ** 2).sum(dim=-1, keepdim=True).T
    
    # ⟨x_i, y_j⟩: (n, m)
    XY = X @ Y.T
    
    # D²[i,j] = ||x_i||² + ||y_j||² - 2⟨x_i, y_j⟩
    D_sq = X_sqnorm + Y_sqnorm - 2 * XY
    
    # Clamp to avoid numerical issues (small negative values)
    D_sq = torch.clamp(D_sq, min=0.0)
    
    return D_sq


def compute_euclidean_distance_matrix(
    X: torch.Tensor, 
    Y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Euclidean distance matrix between X and Y.
    
    Args:
        X: (n, d) tensor of query vectors
        Y: (m, d) tensor of key vectors. If None, computes self-distances.
        
    Returns:
        D: (n, m) distance matrix
    """
    D_sq = compute_squared_euclidean_distance_matrix(X, Y)
    return torch.sqrt(D_sq)


# Complexity analysis
def exact_distance_flops(n: int, m: int, d: int) -> int:
    """
    FLOPs for exact distance matrix computation.
    
    - X @ Y.T: 2*n*m*d
    - ||x||²: n*d
    - ||y||²: m*d  
    - Broadcasting & subtraction: 2*n*m
    
    Total: O(n*m*d) - quadratic in sequence length
    """
    return 2 * n * m * d + n * d + m * d + 2 * n * m

