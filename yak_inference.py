"""Self-contained inference API for the Yak candidate-label classifier."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import PretrainedConfig, PreTrainedModel


_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
_APOS = re.compile(r"(?<=\w)'(?=\w)")
_PUNCT = re.compile(r"([.,!?;:()\"\[\]{}$%&*+/<=>@^_`|~\\-])")
_SPACES = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = _APOS.sub(" '", text)
    text = _PUNCT.sub(r" \1 ", text)
    return _SPACES.sub(" ", text).strip()


def _normalize_label(label: str) -> str:
    return _SPACES.sub(" ", label.replace("_", " ").replace("-", " ")).strip()


class _FastBPE:
    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.tok2id = data["tok2id"]
        self.rank = {tuple(pair): rank for rank, pair in enumerate(data["merges"])}
        self.unk_id = self.tok2id["[UNK]"]
        self.pad_id = self.tok2id["[PAD]"]
        self.cls_id = self.tok2id["[CLS]"]
        self.sep_id = self.tok2id["[SEP]"]
        self.vocab_size = len(self.tok2id)
        self._word_cache: dict[str, list[str]] = {}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in _SPECIALS) + ")")

    def _encode_word(self, word: str) -> list[str]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        symbols = list(word) + ["</w>"]
        last_rank = -1
        while len(symbols) > 1:
            best_pair = None
            best_rank = None
            for index in range(len(symbols) - 1):
                rank = self.rank.get((symbols[index], symbols[index + 1]))
                if rank is not None and rank > last_rank and (best_rank is None or rank < best_rank):
                    best_pair = (symbols[index], symbols[index + 1])
                    best_rank = rank
            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            output = []
            index = 0
            while index < len(symbols):
                if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == best_pair:
                    output.append(merged)
                    index += 2
                else:
                    output.append(symbols[index])
                    index += 1
            symbols = output
            last_rank = best_rank

        self._word_cache[word] = symbols
        return symbols

    def encode_ids(self, text: str, normalize: bool = True) -> list[int]:
        if normalize:
            text = " ".join(
                part if part in _SPECIALS else _normalize_text(part)
                for part in self._special_re.split(text)
            )

        tokens = []
        for part in self._special_re.split(text):
            if part in _SPECIALS:
                tokens.append(part)
            else:
                for word in part.split():
                    tokens.extend(self._encode_word(word))
        return [self.tok2id.get(token, self.unk_id) for token in tokens]


class YakHFConfig(PretrainedConfig):
    model_type = "yak-candidate-classifier"

    def __init__(
        self,
        vocab_size: int = 30005,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.layers = num_layers
        self.max_seq = max_seq_len
        self.pad_id = pad_token_id


class _EmbedPos(nn.Module):
    def __init__(self, config: YakHFConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_len = config.max_seq_len
        self.pad_id = config.pad_token_id
        self.dropout = nn.Dropout(config.dropout)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=config.d_model ** -0.5)
        with torch.no_grad():
            self.tok_emb.weight[config.pad_token_id].zero_()

        position = torch.arange(config.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model)
        )
        pos_emb = torch.zeros(config.max_seq_len, config.d_model)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb.unsqueeze(0), persistent=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_len {self.max_len}")
        token_emb = self.tok_emb(ids) * math.sqrt(self.d_model)
        return self.dropout(token_emb + self.pos_emb[:, :seq_len, :].to(token_emb.dtype))


class _BiDirectionalAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.dropout_p = dropout
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = None if padding_mask is None else padding_mask.bool()[:, None, None, :]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(output)


class _FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_model * 4
        self.linear1 = nn.Linear(d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.attention = _BiDirectionalAttention(d_model, nhead, dropout)
        self.ffn = _FFN(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), padding_mask)
        return x + self.ffn(self.norm2(x))


class YakHFForCandidateClassification(PreTrainedModel):
    config_class = YakHFConfig
    base_model_prefix = "yak"
    all_tied_weights_keys = {}

    def __init__(self, config: YakHFConfig):
        super().__init__(config)
        self.embed = _EmbedPos(config)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(config.d_model, config.nhead, config.dropout) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.scorer = nn.Linear(config.d_model, 1)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        label_positions: torch.Tensor,
        label_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        keep = input_ids != self.config.pad_token_id
        padding_mask = None if bool(keep.all()) else keep
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = self.final_norm(hidden)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device).unsqueeze(1)
        candidate_hidden = hidden[batch_indices, label_positions]
        scores = self.scorer(candidate_hidden).squeeze(-1)
        return scores.masked_fill(~label_mask.bool(), float("-inf"))


class YakPacker:
    def __init__(
        self,
        fast_bpe: _FastBPE,
        class_names: Sequence[str],
        max_labels: int,
        max_label_tokens: int = 12,
        aliases: dict[str, str] | None = None,
    ):
        self.bpe = fast_bpe
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.max_labels = min(max_labels, self.num_classes)
        self.max_text_tokens = 96
        self.max_premise_tokens = 64
        self.max_hypothesis_tokens = 32
        aliases = {"oos": "out of scope"} if aliases is None else aliases
        self.label_ids = []
        for name in self.class_names:
            label_ids = fast_bpe.encode_ids(_normalize_label(aliases.get(name, name)), normalize=True)
            self.label_ids.append(label_ids[:max_label_tokens] or [fast_bpe.unk_id])

    def encode_text(self, text: str) -> list[int]:
        return self.bpe.encode_ids(text, normalize=True)[:self.max_text_tokens]

    def encode_pair(self, premise: str, hypothesis: str) -> list[int]:
        premise_ids = self.bpe.encode_ids(premise, normalize=True)[:self.max_premise_tokens]
        hypothesis_ids = self.bpe.encode_ids(hypothesis, normalize=True)[:self.max_hypothesis_tokens]
        return premise_ids + [self.bpe.sep_id] + hypothesis_ids

    def pack(self, text_ids: Sequence[int], class_order: Sequence[int]) -> tuple[list[int], list[int]]:
        ids = [self.bpe.cls_id] + list(text_ids) + [self.bpe.sep_id]
        label_positions = []
        for class_id in class_order:
            label_positions.append(len(ids))
            ids.extend(self.label_ids[class_id])
            ids.append(self.bpe.sep_id)
        return ids, label_positions


class YakClassifier:
    """Load a published Yak package and predict among caller-supplied candidate labels."""

    def __init__(self, model: YakHFForCandidateClassification, fast_bpe: _FastBPE, device: str | torch.device):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.fast_bpe = fast_bpe
        if fast_bpe.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"Tokenizer has {fast_bpe.vocab_size} tokens but model expects {model.config.vocab_size}."
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        device: str | torch.device | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "YakClassifier":
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            model_dir = local_path
        else:
            model_dir = Path(
                snapshot_download(
                    repo_id=str(model_name_or_path),
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )

        config = YakHFConfig.from_pretrained(model_dir)
        model = YakHFForCandidateClassification.from_pretrained(model_dir, config=config)
        fast_bpe = _FastBPE(model_dir / "bpe.json")
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, fast_bpe, selected_device)

    @torch.inference_mode()
    def predict(
        self,
        text: str | tuple[str, str] | None = None,
        candidate_labels: Sequence[str] | None = None,
        *,
        premise: str | None = None,
        hypothesis: str | None = None,
        n_orders: int = 4,
        seed: int = 0,
        max_label_tokens: int = 12,
    ) -> list[dict[str, float | str]]:
        if candidate_labels is None or len(candidate_labels) == 0:
            raise ValueError("candidate_labels must contain at least one label.")
        if isinstance(candidate_labels, str):
            raise TypeError("candidate_labels must be a sequence of labels, not a string.")
        if n_orders < 1:
            raise ValueError("n_orders must be at least 1.")
        if max_label_tokens < 1:
            raise ValueError("max_label_tokens must be at least 1.")

        pair = None
        if isinstance(text, tuple):
            if len(text) != 2 or premise is not None or hypothesis is not None:
                raise ValueError("Pass a two-item text tuple or premise and hypothesis, not both.")
            pair = text
        elif premise is not None or hypothesis is not None:
            if premise is None or hypothesis is None or text is not None:
                raise ValueError("Provide both premise and hypothesis, without text.")
            pair = (premise, hypothesis)
        elif not isinstance(text, str):
            raise ValueError("Provide text, or both premise and hypothesis.")

        packer = YakPacker(self.fast_bpe, candidate_labels, len(candidate_labels), max_label_tokens)
        text_ids = packer.encode_pair(*pair) if pair is not None else packer.encode_text(text)
        probabilities = np.zeros(len(candidate_labels), dtype=np.float64)

        for order_index in range(n_orders):
            order = list(range(len(candidate_labels)))
            if order_index > 0:
                random.Random(seed + order_index).shuffle(order)
            input_values, positions = packer.pack(text_ids, order)
            if len(input_values) > self.model.config.max_seq_len:
                raise ValueError(
                    f"Packed input has {len(input_values)} tokens, exceeding "
                    f"max_seq_len={self.model.config.max_seq_len}. Reduce the text length "
                    "or candidate-label count."
                )

            input_ids = torch.tensor([input_values], dtype=torch.long, device=self.device)
            label_positions = torch.tensor([positions], dtype=torch.long, device=self.device)
            label_mask = torch.ones_like(label_positions, dtype=torch.bool)
            scores = self.model(
                input_ids=input_ids,
                label_positions=label_positions,
                label_mask=label_mask,
            ).float().cpu().numpy().astype(np.float64)
            scores -= scores.max(axis=1, keepdims=True)
            slot_probabilities = np.exp(scores)
            slot_probabilities /= slot_probabilities.sum(axis=1, keepdims=True)
            for slot, class_id in enumerate(order):
                probabilities[class_id] += slot_probabilities[0, slot]

        probabilities /= n_orders
        ranked = sorted(
            zip(candidate_labels, probabilities.tolist()),
            key=lambda item: -item[1],
        )
        return [{"label": label, "score": score} for label, score in ranked]


__all__ = ["YakClassifier", "YakHFConfig", "YakHFForCandidateClassification", "YakPacker"]
