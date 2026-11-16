import math
import torch
from torch import nn


class EmbedPos(nn.Module):
    """Token embedding + static sinusoidal positions (sin/cos), followed by dropout.

    Token embeddings are initialised with std = d_model ** -0.5 and scaled by
    sqrt(d_model) in forward, so they have unit scale and are not drowned out by the
    +-1 positional signal. The same init keeps the tied output head's starting
    logits small, so the initial loss lands near ln(vocab_size).

    Args:
        vocab_size: number of token ids (len(tok2id), including special tokens)
        d_model: model width (must be even)
        max_len: length of the sinusoidal table
        pad_id: id of the padding token
        dropout: dropout rate applied to the summed embedding
    """

    def __init__(self, vocab_size, d_model=256, max_len=2048, pad_id=0, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sin/cos pairs"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.dropout = nn.Dropout(dropout)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.tok_emb.weight[pad_id].zero_()
        # Fixed table: moves with .to(device), is not a parameter, and is not saved in checkpoints.
        self.register_buffer(
            "pos_emb", self._get_positional_encoding(max_len, d_model), persistent=False
        )

    @staticmethod
    def _get_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, ids):
        # ids: (batch_size, seq_len)
        seq_len = ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_len {self.max_len}")
        tok_emb = self.tok_emb(ids) * math.sqrt(self.d_model)  # (B, L, d_model)
        emb = tok_emb + self.pos_emb[:, :seq_len, :].to(tok_emb.dtype)
        return self.dropout(emb)