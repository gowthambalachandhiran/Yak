import torch
import torch.nn as nn
import math

class BiDirectionalAttention(nn.Module):
    """Multi head attention self-attention module
    Args:
        d_model: dimension of the model
        nhead: number of heads
        dropout: dropout rate
    """
    def __init__(self,d_model:int,nhead:int,dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,padding_mask:torch.Tensor| None = None) -> torch.Tensor:
        """Forward pass of the multi head attention module
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            padding_mask: optional mask tensor of shape (batch_size, seq_len)
        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).view(batch_size,seq_len,3,self.nhead,self.d_k)
        qkv = qkv.permute(2,0,3,1,4)  # (3, batch_size, nhead, seq_len, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (batch_size, nhead, seq_len, d_k)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, nhead, seq_len, seq_len)
        if padding_mask is not None:
            # Expand padding mask to match attention scores shape
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(padding_mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, nhead, seq_len, seq_len)
        attn_weights = self.attn_dropout(attn_weights)

        # [batch_size, nhead, seq_len, d_k] = [batch_size, nhead, seq_len, seq_len] @ [batch_size, nhead, seq_len, d_k
        output = torch.matmul(attn_weights, v)  # (batch_size, nhead, seq_len, d_k)

        #concat heads and project
        context = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.out_proj(context)  # (batch_size, seq_len, d_model)
        return output