#Mixture.py
"""Train on several datasets at once. (Packer and ClsDataset live in the notebook and are passed in.)

Each dataset keeps its own label set and its own Packer. A training batch is drawn from ONE dataset (chosen at
random with the given weights), so lengths in a batch are similar and candidate labels never leak across tasks.
Inside a dataset, label order and label subsets are reshuffled every time an example is drawn.

No torch import here, so everything can be tested without torch.
"""
import json
import os
import random



def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_train_val(rows, val_frac=0.1, seed=0):
    """Banking77 has no validation split: hold out a fixed-seed fraction of train, per class."""
    by_class = {}
    for i, r in enumerate(rows):
        by_class.setdefault(r["label"], []).append(i)
    rng = random.Random(seed)
    val_idx = set()
    for idxs in by_class.values():
        rng.shuffle(idxs)
        val_idx.update(idxs[: max(1, int(round(len(idxs) * val_frac)))])
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    return train, val


def load_task(name, data_dir, fast_bpe, packer_cls, max_labels=160, val_frac=0.1, seed=0, **packer_kw):
    """Read Data/<name>/{train,validation,test}.jsonl and labels.json. Returns None if the dataset is missing.

    Datasets without a validation file (Banking77) get one cut from train with a fixed seed.
    """
    d = os.path.join(data_dir, name)
    if not os.path.exists(os.path.join(d, "labels.json")):
        return None
    meta = json.load(open(os.path.join(d, "labels.json"), encoding="utf-8"))
    train = read_jsonl(os.path.join(d, "train.jsonl"))
    test = read_jsonl(os.path.join(d, "test.jsonl"))
    val_path = os.path.join(d, "validation.jsonl")
    if os.path.exists(val_path):
        val = read_jsonl(val_path)
    else:
        train, val = split_train_val(train, val_frac, seed)
    packer = packer_cls(fast_bpe, meta["names"], max_labels=max_labels, **packer_kw)
    return {"name": name, "names": meta["names"], "oos_index": meta.get("oos_index"),
            "train": train, "val": val, "test": test, "packer": packer}


def _to_row(rec, label_id):
    """One processed record -> the row format ClsDataset reads (text, or premise and hypothesis)."""
    if "premise" in rec and "hypothesis" in rec:
        return {"premise": rec["premise"], "hypothesis": rec["hypothesis"], "label": label_id}
    text = rec["text"]
    if text.startswith("Premise:") and "\nHypothesis:" in text:      # older single-string form of a pair
        premise, hypothesis = text[len("Premise:"):].split("\nHypothesis:", 1)
        return {"premise": premise.strip(), "hypothesis": hypothesis.strip(), "label": label_id}
    return {"text": text, "label": label_id}


def load_processed(processed_dir, fast_bpe, packer_cls, datasets=None, max_labels=160, **packer_kw):
    """Load the output of the preprocessing script: train/validation/test.jsonl with one record per example
    (source_dataset, text or premise+hypothesis, label as a name). The splits are used as they are, so
    any balancing done by the script is kept.

    Class ids come from label_metadata.json (label_vocabularies) when it exists, else from the training labels.
    Returns a list of task dicts like load_task, in the order given by `datasets` (default: order of appearance).
    """
    meta_path = os.path.join(processed_dir, "label_metadata.json")
    vocabs = json.load(open(meta_path, encoding="utf-8")).get("label_vocabularies") if os.path.exists(meta_path) else None

    recs = {}                                     # dataset -> split -> records
    for fname, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        path = os.path.join(processed_dir, fname + ".jsonl")
        if not os.path.exists(path):
            continue
        for rec in read_jsonl(path):
            recs.setdefault(rec["source_dataset"], {}).setdefault(split, []).append(rec)
    if not recs:
        raise FileNotFoundError(f"no train/validation/test.jsonl found in {processed_dir}")

    tasks = []
    for name in (datasets or list(recs)):
        if name not in recs:
            print(f"skipping {name}: no records in {processed_dir}")
            continue
        by_split = recs[name]
        names = list(vocabs[name]) if vocabs and name in vocabs else sorted({r["label"] for r in by_split["train"]})
        if all(n.strip().lstrip("-").isdigit() for n in names):
            raise ValueError(f"{name}: labels look like numeric ids ({names[:5]}). The label text is what the model "
                             f"reads, so the preprocessing must write label names (e.g. card_arrival), not ids.")
        index = {n: i for i, n in enumerate(names)}
        splits = {}
        for split, rs in by_split.items():
            unknown = {r["label"] for r in rs if r["label"] not in index}
            if unknown:
                raise ValueError(f"{name}/{split}: labels not in the vocabulary: {sorted(unknown)[:5]}")
            splits[split] = [_to_row(r, index[r["label"]]) for r in rs]
        if "train" not in splits or "val" not in splits or "test" not in splits:
            raise ValueError(f"{name}: needs train, validation and test records, found {sorted(splits)}")
        tasks.append({"name": name, "names": names, "oos_index": index.get("oos"),
                      "train": splits["train"], "val": splits["val"], "test": splits["test"],
                      "packer": packer_cls(fast_bpe, names, max_labels=max_labels, **packer_kw)})
    return tasks


def pick_holdout(task, k, seed=0):
    """k classes of the task to keep out of training (never the out-of-scope class)."""
    if not k:
        return []
    ids = [c for c in range(len(task["names"])) if c != task.get("oos_index")]
    return sorted(random.Random(seed).sample(ids, min(k, len(ids) - 2)))


def apply_holdout(task, held_out):
    """Remove held-out classes from the training rows and from the packer's training candidates.
    val/test keep every class so seen and unseen labels can be measured separately."""
    held = set(held_out)
    task["held_out"] = sorted(held)
    task["train_all"] = task["train"]
    task["train"] = [r for r in task["train"] if r["label"] not in held]
    task["packer"].set_train_classes([c for c in range(len(task["names"])) if c not in held])
    task["seen_ids"] = [c for c in range(len(task["names"])) if c not in held]
    return task


def split_seen_unseen(rows, held_out):
    held = set(held_out)
    return [r for r in rows if r["label"] not in held], [r for r in rows if r["label"] in held]


def mixture_weights(sizes, mode="equal"):
    """Probability of drawing the next batch from each dataset.
    mode: "equal", "sqrt" (proportional to sqrt of size), "size", or a list of explicit weights."""
    if isinstance(mode, (list, tuple)):
        w = list(mode)
    elif mode == "equal":
        w = [1.0] * len(sizes)
    elif mode == "sqrt":
        w = [s ** 0.5 for s in sizes]
    elif mode == "size":
        w = [float(s) for s in sizes]
    else:
        raise ValueError(mode)
    total = sum(w)
    return [x / total for x in w]


class MixtureBatchSampler:
    """Yields batches of (task, local_index, nonce). Pass it to DataLoader(batch_sampler=...).

    Each dataset is walked in a shuffled order and restarted when exhausted, so small datasets repeat
    (with new label orders each time) while large ones are not exhausted in an epoch.
    """

    def __init__(self, sizes, weights, batch_sizes, steps_per_epoch, seed=0):
        assert len(sizes) == len(weights) == len(batch_sizes)
        self.sizes, self.weights, self.batch_sizes = sizes, weights, batch_sizes
        self.steps_per_epoch, self.seed, self.epoch = steps_per_epoch, seed, 0
        self.last_task_counts = [0] * len(sizes)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        perms = [rng.sample(range(n), n) for n in self.sizes]
        pos = [0] * len(self.sizes)
        counts = [0] * len(self.sizes)
        nonce = (self.epoch + 1) * 10_000_000
        for _ in range(self.steps_per_epoch):
            t = rng.choices(range(len(self.sizes)), weights=self.weights)[0]
            batch = []
            for _ in range(self.batch_sizes[t]):
                if pos[t] == len(perms[t]):
                    perms[t] = rng.sample(range(self.sizes[t]), self.sizes[t])
                    pos[t] = 0
                batch.append((t, perms[t][pos[t]], nonce))
                pos[t] += 1
                nonce += 1
            counts[t] += 1
            self.last_task_counts = counts
            yield batch


class MixedTrainDataset:
    """Holds one training dataset per task. Items are the (task, local_index, nonce) tuples the sampler yields.
    dataset_cls is the notebook's ClsDataset (it must offer .example(index, nonce))."""

    def __init__(self, tasks, dataset_cls, seed=0, full_prob=0.5, min_labels=2):
        self.tasks = tasks
        self.parts = [dataset_cls(t["train"], t["packer"], train=True, seed=seed, full_prob=full_prob,
                                 min_labels=min_labels, task_id=i) for i, t in enumerate(tasks)]

    def sizes(self):
        return [len(p) for p in self.parts]

    def __len__(self):
        return sum(self.sizes())

    def __getitem__(self, item):
        t, local, nonce = item
        return self.parts[t].example(local, nonce)