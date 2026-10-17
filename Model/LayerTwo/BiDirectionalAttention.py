import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDirectionalAttention(nn.Module):
    """Multi head bidirectional self-attention.

    Same interface and maths as before, but uses F.scaled_dot_product_attention so the
    (seq_len x seq_len) score matrix is never fully materialised. That matters at the
    2,048 token context.

    Args:
        d_model: dimension of the model
        nhead: number of heads
        dropout: attention dropout rate
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            padding_mask: optional (batch_size, seq_len), True/1 for real tokens, False/0 for padding
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nhead, L, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.bool()[:, None, None, :]  # (B, 1, 1, L), True = may attend

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B, nhead, L, d_k)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(out)