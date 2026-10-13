import math
import torch
from torch import nn

class EmbedPos(nn.Module):
    def __init__(self,vocab_size,d_model=128,max_len=2048,pad_id=0,dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.dropout = nn.Dropout(dropout)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.register_buffer('pos_emb', self._get_positional_encoding(max_len, d_model))

    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe

    def forward(self, ids):
        # ids: (batch_size, seq_len)
        tok_emb = self.tok_emb(ids)  # (batch_size, seq_len, d_model)
        seq_len = ids.size(1)
        pos_emb = self.pos_emb[:, :seq_len, :]  # (1, seq_len, d_model)
        emb = tok_emb + pos_emb  # (batch_size, seq_len, d_model)
        emb = self.dropout(emb)
        return emb