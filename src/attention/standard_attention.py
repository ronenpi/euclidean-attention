"""
Standard Dot-Product Attention (Baseline)

Attention(Q, K, V) = softmax(QK^T / √d) @ V

This is the O(n²) baseline we're trying to improve upon.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class StandardAttention(nn.Module):
    """
    Standard scaled dot-product attention.
    
    Complexity: O(n² * d) time, O(n²) space for attention matrix
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
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
            attn_mask: (seq_len, seq_len) or (batch, seq_len, seq_len)
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
        Q = self.q_proj(query)  # (batch, seq_len, embed_dim)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
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
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, V)
        
        # Reshape back
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


def compute_attention_flops(batch_size: int, seq_len: int, embed_dim: int, num_heads: int) -> dict:
    """
    Compute FLOPs breakdown for standard attention.
    """
    head_dim = embed_dim // num_heads
    
    # Q, K, V projections: 3 * (batch * seq * embed * embed) * 2 (mul + add)
    proj_flops = 3 * batch_size * seq_len * embed_dim * embed_dim * 2
    
    # QK^T: batch * heads * seq * seq * head_dim * 2
    qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2
    
    # Softmax: ~5 * batch * heads * seq * seq (exp, sum, div, etc.)
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    
    # Attention @ V: batch * heads * seq * seq * head_dim * 2
    av_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2
    
    # Output projection: batch * seq * embed * embed * 2
    out_flops = batch_size * seq_len * embed_dim * embed_dim * 2
    
    total = proj_flops + qk_flops + softmax_flops + av_flops + out_flops
    
    return {
        'projections': proj_flops,
        'qk_matmul': qk_flops,
        'softmax': softmax_flops,
        'av_matmul': av_flops,
        'output_proj': out_flops,
        'total': total,
        'quadratic_component': qk_flops + softmax_flops + av_flops,
    }

