#Infer.py
"""Classify one text against labels the user supplies at call time. packer_cls is the notebook's Packer."""
import random

import numpy as np
import torch

from .Metrics import softmax


@torch.no_grad()
def classify(model, fast_bpe, text, labels, packer_cls, device="cpu", n_orders=4, seed=0, max_label_tokens=12):
    """Return [(label, probability), ...] sorted from most to least likely.

    The labels can be any strings. Probabilities are averaged over n_orders shuffles of the label list,
    because the encoder is only approximately order independent.
    """
    model.eval()
    pk = packer_cls(fast_bpe, labels, max_labels=len(labels), max_label_tokens=max_label_tokens, aliases={})
    # text is a string, or a (premise, hypothesis) tuple for sentence-pair tasks
    text_ids = pk.encode_pair(*text) if isinstance(text, tuple) else pk.encode_text(text)
    n = len(labels)
    probs = np.zeros(n)
    for k in range(n_orders):
        order = list(range(n))
        if k > 0:
            random.Random(seed + k).shuffle(order)
        ids, pos = pk.pack(text_ids, order)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        pos_t = torch.tensor([pos], dtype=torch.long, device=device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        scores = model(ids_t, pos_t, mask).float().cpu().numpy()
        p = softmax(scores)[0]
        for slot, cls in enumerate(order):
            probs[cls] += p[slot]
    probs /= n_orders
    return sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])