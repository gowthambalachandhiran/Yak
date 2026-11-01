import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.LayerOne.EmbedPostional import EmbedPos
from Model.EncoderBlock import EncoderBlock


class MlmModel(nn.Module):
    """Masked language model, phase 1 pre-training.

    ids -> EmbedPos -> N x EncoderBlock (pre-norm) -> final LayerNorm -> tied MLM head -> logits

    The head shares its weight with the token embedding, plus a separate output bias.

    Args:
        vocab_size: number of token ids, INCLUDING special tokens (len(tokenizer.tok2id))
        d_model: model width
        nhead: attention heads
        num_layers: number of encoder blocks
        max_seq_len: length of the sinusoidal positional table
        dropout: dropout rate
        pad_token_id: id of the padding token
    """

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
                 max_seq_len: int, dropout: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.pad_id = pad_token_id
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = EmbedPos(vocab_size, d_model=d_model, max_len=max_seq_len,
                              pad_id=pad_token_id, dropout=dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        # Blocks are pre-norm, so the last residual stream must be normalised before the head.
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head_bias = nn.Parameter(torch.zeros(vocab_size))

        self.apply(self._init_weights)
        # GPT-2 style: shrink the init of the two residual output projections per block.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("linear2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

        emb_weight = self.embed.tok_emb.weight
        assert emb_weight.shape == (vocab_size, d_model), (
            f"Embedding weight shape {tuple(emb_weight.shape)} != {(vocab_size, d_model)}"
        )
        self.head.weight = emb_weight  # tie head to the token embedding

    def _init_weights(self, m):
        # The head is skipped: it is tied to the embedding, which EmbedPos already initialised.
        if isinstance(m, nn.Linear) and m is not self.head:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weight is counted once
        if exclude_embedding:
            n -= self.embed.tok_emb.weight.numel()
        return n

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) ids -> (batch, seq_len, d_model) hidden states."""
        keep = ids != self.pad_id  # True for real tokens
        # Fully packed batches have no padding; skipping the mask enables the fastest attention kernels.
        padding_mask = None if bool(keep.all()) else keep
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, padding_mask)
        return self.final_norm(x)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.head.weight, self.head_bias)

    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None):
        """
        labels is None -> full logits (batch, seq_len, vocab_size), for inference.
        labels given   -> (loss, logits_at_selected_positions, targets). Only positions with
                          labels != -100 go through the vocabulary projection, which saves a
                          lot of memory (about 15% of positions instead of all of them).
        """
        h = self.encode(ids)
        if labels is None:
            return self._logits(h)
        sel = labels != -100
        targets = labels[sel]
        if targets.numel() == 0:
            return h.sum() * 0.0, None, targets
        logits = self._logits(h[sel])  # (n_selected, vocab_size)
        loss = F.cross_entropy(logits.float(), targets)
        return loss, logits, targets