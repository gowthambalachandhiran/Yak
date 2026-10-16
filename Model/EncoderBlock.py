import torch 
import torch.nn as nn


from Model.LayerTwo.BiDirectionalAttention import BiDirectionalAttention
from Model.LayerThree.FFN import FFN

class EncoderBlock(nn.Module):
    """Pre norm encoder block with multi head attention and feed forward network
    x = x+attebtion(LayerNorm(x),mask)
    x = x+FFN(LayerNorm(x))

    Args:
        d_model: dimension of the model
        nhead: number of heads
        dropout: dropout rate
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    def __init__(self,d_model:int,nhead:int,dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        self.attention = BiDirectionalAttention(d_model,nhead,dropout)
        self.ffn = FFN(d_model,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x:torch.Tensor,padding_mask:torch.Tensor| None = None) -> torch.Tensor:
        """Feed forward pass of the encoder block
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            padding_mask: optional mask tensor of shape (batch_size, seq_len)
        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        # Sub 1: Normalize -> Attention (mask goes here only for the attention) -> Skip connection
        x_norm = self.norm1(x)
        attn_output = self.attention(x_norm,padding_mask)
        x = x + attn_output

        # Sub 2: Normalize -> Feed Forward -> Skip connection
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        return x
    


