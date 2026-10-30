#LabelPack.py
"""Pack a text and a list of candidate labels into one sequence for the label-scoring classifier.

    [CLS] text tokens [SEP] label1 tokens [SEP] label2 tokens [SEP] ... labelN tokens [SEP]

label_pos[m] is the index of the FIRST token of the m-th label in that sequence. The model gathers the
encoder output at those positions, scores each with one shared linear layer, and softmaxes over the slots.
A slot is a position in this example's own label list, not a fixed class, so the target is "the slot where
the true class ended up".

Training builds a different label list every time an example is drawn (random subset, random order, true
label always included). Evaluation uses a fixed list.

No torch import at module level (make_collate imports it lazily), so the packing logic can be tested without torch.
"""
import random

from Utils.FastBpe import normalize_text, normalize_label


LABEL_ALIASES = {"oos": "out of scope"}   # names that carry no meaning as words


class Packer:
    def __init__(self, fast_bpe, class_names, max_labels=160, max_text_tokens=96,
                 max_label_tokens=12, aliases=None, max_premise_tokens=64, max_hypothesis_tokens=32):
        """
        fast_bpe:        FastBPE
        class_names:     list of class names, the position is the class id
        max_labels:      upper bound M on labels in one packed example
        max_text_tokens: the text is cut to this many tokens
        max_label_tokens: each label is cut to this many tokens
        """
        self.bpe = fast_bpe
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.max_labels = min(max_labels, self.num_classes)
        self.train_ids = list(range(self.num_classes))     # classes that may appear as training candidates
        self.max_text_tokens = max_text_tokens
        self.max_premise_tokens = max_premise_tokens
        self.max_hypothesis_tokens = max_hypothesis_tokens
        aliases = LABEL_ALIASES if aliases is None else aliases
        self.label_ids = []
        for name in self.class_names:
            ids = fast_bpe.encode_ids(normalize_label(aliases.get(name, name)), normalize=True)
            ids = ids[:max_label_tokens] or [fast_bpe.unk_id]
            self.label_ids.append(ids)

    # ------------------------------------------------------------------ text
    def encode_text(self, text):
        return self.bpe.encode_ids(text, normalize=True)[: self.max_text_tokens]

    def encode_pair(self, premise, hypothesis):
        """Sentence pair as one 'text': premise tokens [SEP] hypothesis tokens, each cut separately.
        The model has no segment embeddings, so the [SEP] between them is the only divider."""
        p = self.bpe.encode_ids(premise, normalize=True)[: self.max_premise_tokens]
        h = self.bpe.encode_ids(hypothesis, normalize=True)[: self.max_hypothesis_tokens]
        return p + [self.bpe.sep_id] + h

    def encode_row(self, row):
        if "premise" in row:
            return self.encode_pair(row["premise"], row["hypothesis"])
        return self.encode_text(row["text"])

    def max_text_len(self):
        return max(self.max_text_tokens, self.max_premise_tokens + 1 + self.max_hypothesis_tokens)

    # ------------------------------------------------------------------ packing
    def pack(self, text_ids, class_order):
        """Return (ids, label_pos) for the given text ids and classes in the given order."""
        ids = [self.bpe.cls_id] + list(text_ids) + [self.bpe.sep_id]
        label_pos = []
        for c in class_order:
            label_pos.append(len(ids))
            ids.extend(self.label_ids[c])
            ids.append(self.bpe.sep_id)
        return ids, label_pos

    def max_packed_len(self):
        longest = sorted((len(x) + 1 for x in self.label_ids), reverse=True)[: self.max_labels]
        return 2 + self.max_text_len() + sum(longest)

    def set_train_classes(self, class_ids):
        """Restrict training candidates to class_ids (the rest are held out and never shown in training).
        Evaluation still uses every class."""
        self.train_ids = sorted(class_ids)

    # ------------------------------------------------------------------ choosing the label list
    def train_labels(self, true_class, rng, full_prob=0.5, min_labels=2):
        """Random label list that always contains true_class.

        With probability full_prob use as many labels as allowed (so the model sees the real test
        condition), otherwise draw the count uniformly from min_labels..max_labels.
        """
        assert true_class in self.train_ids, f"class {true_class} is held out and cannot be a training target"
        cap = min(self.max_labels, len(self.train_ids))
        lo = min(min_labels, cap)
        n = cap if rng.random() < full_prob else rng.randint(lo, cap)
        others = [c for c in self.train_ids if c != true_class]
        chosen = rng.sample(others, n - 1) + [true_class]
        rng.shuffle(chosen)
        return chosen

    def eval_labels(self, true_class, rng=None):
        """All classes (up to max_labels). If there are more classes than max_labels, keep the true
        label plus a fixed-seed random subset. Order is class-id order unless rng shuffles it."""
        order = list(range(self.num_classes))
        if self.num_classes > self.max_labels:
            sub_rng = random.Random(true_class)
            others = [c for c in order if c != true_class]
            order = sorted(sub_rng.sample(others, self.max_labels - 1) + [true_class])
        if rng is not None:
            rng.shuffle(order)
        return order


class ClsDataset:
    """Map-style dataset (works with torch DataLoader, no torch import needed here).

    rows:   list of dicts with "text" and "label" (the jsonl format written by Data/*.py)
    train:  True  -> random label subsets and order, different every epoch (call set_epoch)
            False -> all labels in class order
    """

    def __init__(self, rows, packer, train, seed=0, full_prob=0.5, min_labels=2, eval_shuffle_seed=None, task_id=0):
        self.packer = packer
        self.task_id = task_id
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.full_prob = full_prob
        self.min_labels = min_labels
        self.eval_shuffle_seed = eval_shuffle_seed   # None = class-id order; int = a fixed shuffled order
        self.text_ids = [packer.encode_row(r) for r in rows]      # single texts or premise/hypothesis pairs
        self.labels = [int(r["label"]) for r in rows]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.example(idx, self.epoch * 10_007 + idx)

    def example(self, idx, nonce):
        """Example idx with a label list drawn from `nonce` (a different nonce gives a different list)."""
        true_class = self.labels[idx]
        if self.train:
            rng = random.Random(self.seed * 1_000_003 + nonce)
            order = self.packer.train_labels(true_class, rng, self.full_prob, self.min_labels)
        else:
            rng = None
            if self.eval_shuffle_seed is not None:
                rng = random.Random(self.eval_shuffle_seed * 1_000_003 + idx)
            order = self.packer.eval_labels(true_class, rng)
        ids, label_pos = self.packer.pack(self.text_ids[idx], order)
        return {"ids": ids, "label_pos": label_pos, "slot_class": order,
                "target": order.index(true_class), "true_class": true_class, "task_id": self.task_id}


def collate_lists(batch, pad_id):
    """Pad a list of examples to the longest sequence and the largest label count in the batch.

    Returns plain python lists so it can be tested without torch:
        ids [B, L] (pad_id), label_pos [B, M] (0 for unused), label_mask [B, M] (True = real label),
        slot_class [B, M] (-1 for unused), target [B], true_class [B]
    """
    L = max(len(b["ids"]) for b in batch)
    M = max(len(b["label_pos"]) for b in batch)
    out = {"ids": [], "label_pos": [], "label_mask": [], "slot_class": [], "target": [], "true_class": [],
           "task_id": []}
    for b in batch:
        n = len(b["label_pos"])
        out["ids"].append(b["ids"] + [pad_id] * (L - len(b["ids"])))
        out["label_pos"].append(b["label_pos"] + [0] * (M - n))
        out["label_mask"].append([True] * n + [False] * (M - n))
        out["slot_class"].append(b["slot_class"] + [-1] * (M - n))
        out["target"].append(b["target"])
        out["true_class"].append(b["true_class"])
        out["task_id"].append(b.get("task_id", 0))
    return out


def make_collate(pad_id):
    """Collate function for torch DataLoader: python lists -> tensors."""
    def collate(batch):
        import torch
        d = collate_lists(batch, pad_id)
        return {
            "ids": torch.tensor(d["ids"], dtype=torch.long),
            "label_pos": torch.tensor(d["label_pos"], dtype=torch.long),
            "label_mask": torch.tensor(d["label_mask"], dtype=torch.bool),
            "slot_class": torch.tensor(d["slot_class"], dtype=torch.long),
            "target": torch.tensor(d["target"], dtype=torch.long),
            "true_class": torch.tensor(d["true_class"], dtype=torch.long),
            "task_id": torch.tensor(d["task_id"], dtype=torch.long),
        }
    return collate