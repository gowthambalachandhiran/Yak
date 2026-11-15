import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .configuration_yak import YakConfig


class YakEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        if d % 2:
            raise ValueError("d_model must be even")
        self.max_len = config.max_seq_len
        self.pad_id = config.pad_token_id
        self.tok_emb = nn.Embedding(config.vocab_size, d, padding_idx=self.pad_id)
        self.dropout = nn.Dropout(config.dropout)
        pos = torch.arange(config.max_seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
        pe = torch.zeros(config.max_seq_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_emb", pe.unsqueeze(0), persistent=False)

    def forward(self, ids):
        n = ids.size(1)
        if n > self.max_len:
            raise ValueError(f"sequence length {n} exceeds max_seq_len={self.max_len}")
        x = self.tok_emb(ids) * (self.tok_emb.embedding_dim ** 0.5)
        return self.dropout(x + self.pos_emb[:, :n].to(x.dtype))


class YakAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        if d % config.nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead, self.dk = config.nhead, d // config.nhead
        self.dropout_p = config.dropout
        self.qkv_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, padding_mask=None):
        b, n, d = x.shape
        qkv = self.qkv_proj(x).view(b, n, 3, self.nhead, self.dk).permute(2, 0, 3, 1, 4)
        mask = padding_mask[:, None, None, :] if padding_mask is not None else None
        y = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(y.transpose(1, 2).reshape(b, n, d))


class YakFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.linear1 = nn.Linear(d, 4*d)
        self.linear2 = nn.Linear(4*d, d)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class YakBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.attention = YakAttention(config)
        self.ffn = YakFFN(config)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class YakForLabelScoring(PreTrainedModel):
    config_class = YakConfig
    base_model_prefix = "yak"
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.embed = YakEmbeddings(config)
        self.blocks = nn.ModuleList([YakBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.scorer = nn.Linear(config.d_model, 1)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed.tok_emb

    def set_input_embeddings(self, value):
        self.embed.tok_emb = value

    def encode(self, input_ids):
        keep = input_ids.ne(self.config.pad_token_id)
        mask = None if bool(keep.all()) else keep
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x)

    def score_candidates(self, input_ids, label_positions, label_mask):
        h = self.encode(input_ids)
        idx = label_positions.unsqueeze(-1).expand(-1, -1, h.size(-1))
        selected = torch.gather(h, 1, idx)
        scores = self.scorer(selected).squeeze(-1)
        return scores.masked_fill(~label_mask.bool(), float("-inf"))

    def forward(self, input_ids=None, label_positions=None, label_mask=None,
                labels=None, return_dict=None, **kwargs):
        if input_ids is None or label_positions is None or label_mask is None:
            raise ValueError("input_ids, label_positions and label_mask are required")
        scores = self.score_candidates(input_ids, label_positions, label_mask)
        loss = F.cross_entropy(scores.float(), labels) if labels is not None else None
        if return_dict is False:
            result = (scores,)
            return (loss,) + result if loss is not None else result
        return SequenceClassifierOutput(loss=loss, logits=scores)
