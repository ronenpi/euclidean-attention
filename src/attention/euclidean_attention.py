"""
Euclidean Distance Attention

Instead of dot-product similarity, use negative squared Euclidean distance:

    Attention(Q, K, V) = softmax(-||Q - K||² / τ) @ V

Properties:
- Sensitive to vector magnitudes (not just angles)
- Naturally bounded attention (large distances → small weights)  
- Different inductive bias than dot-product

Complexity: Still O(n²) for exact computation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class EuclideanAttention(nn.Module):
    """
    Euclidean distance-based attention mechanism.
    
    Key differences from standard attention:
    1. Uses squared Euclidean distance instead of dot product
    2. Distance is negated (closer = higher attention)
    3. Temperature parameter τ controls attention sharpness
    
    Relationship: ||q-k||² = ||q||² + ||k||² - 2⟨q,k⟩
    So Euclidean attention captures norm information that dot-product ignores.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        temperature: Optional[float] = None,  # If None, learned per head
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Temperature parameter
        if temperature is not None:
            self.register_buffer('temperature', torch.tensor(temperature))
        elif learnable_temperature:
            # Initialize temperature similar to 1/√d for comparability
            init_temp = self.head_dim
            self.temperature = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))
        else:
            self.register_buffer('temperature', torch.tensor(float(self.head_dim)))
    
    def _compute_squared_distances(
        self,
        Q: torch.Tensor,
        K: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute squared Euclidean distances between Q and K.
        
        ||q - k||² = ||q||² + ||k||² - 2⟨q, k⟩
        
        Args:
            Q: (batch, heads, seq_q, head_dim)
            K: (batch, heads, seq_k, head_dim)
            
        Returns:
            D²: (batch, heads, seq_q, seq_k)
        """
        # ||q||² for each query: (batch, heads, seq_q, 1)
        Q_sqnorm = (Q ** 2).sum(dim=-1, keepdim=True)
        
        # ||k||² for each key: (batch, heads, 1, seq_k)
        K_sqnorm = (K ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        
        # ⟨q, k⟩: (batch, heads, seq_q, seq_k)
        QK = torch.matmul(Q, K.transpose(-2, -1))
        
        # ||q - k||² = ||q||² + ||k||² - 2⟨q, k⟩
        D_sq = Q_sqnorm + K_sqnorm - 2 * QK
        
        # Clamp for numerical stability
        D_sq = torch.clamp(D_sq, min=0.0)
        
        return D_sq
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, seq_len, embed_dim)
            key: (batch, seq_len, embed_dim) - defaults to query
            value: (batch, seq_len, embed_dim) - defaults to key
            attn_mask: (seq_len, seq_len) additive mask
            return_attention: whether to return attention weights
            
        Returns:
            output: (batch, seq_len, embed_dim)
            attn_weights: (batch, num_heads, seq_len, seq_len) if return_attention
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute squared distances
        D_sq = self._compute_squared_distances(Q, K)
        
        # Convert distances to attention scores
        # Attention score = -D² / τ (closer = higher score)
        attn_scores = -D_sq / self.temperature
        
        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores + attn_mask
            
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
            
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class EuclideanSelfAttention(EuclideanAttention):
    """Convenience wrapper for self-attention."""
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return super().forward(x, x, x, attn_mask, return_attention)

