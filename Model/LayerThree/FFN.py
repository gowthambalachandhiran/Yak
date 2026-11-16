
import torch
import torch.nn as nn

class FFN(nn.Module):
    """Feed Forward Network module
    Args:
        d_model: dimension of the model
        d_ff: dimension of the feed forward network
        dropout: dropout rate
    """
    def __init__(self,d_model:int,dropout:float=0.1,ff_multiplier:int=4):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_model * ff_multiplier
        self.linear1 = nn.Linear(d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x