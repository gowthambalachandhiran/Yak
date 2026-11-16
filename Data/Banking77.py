"""Banking77: 77 customer-service intents in the banking domain.

Source: the PolyAI task-specific-datasets repository on GitHub (train.csv and test.csv, columns "text" and "category").
Splits: train and test only. There is no official validation split, so create one from train (with a fixed seed)
in the fine-tuning code.
Label ids: the 77 category names sorted alphabetically.
"""
import csv

try:
    from .BaseData import BaseDataset
    from .FileUtils import fetch, clean
except ImportError:
    from BaseData import BaseDataset
    from FileUtils import fetch, clean

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


class Banking77(BaseDataset):
    name = "banking77"
    expected_splits = {"train": 10003, "test": 3080}
    expected_labels = 77
    sources = {"train": f"{BASE}/train.csv", "test": f"{BASE}/test.csv"}

    def text_of(self, row):
        return row["text"]

    def build(self):
        raw, shas = {}, {}
        for split, url in self.sources.items():
            filename = f"banking77_{split}.csv"
            path, shas[filename] = fetch(url, self.raw_dir, filename)
            with open(path, "r", encoding="utf-8", newline="") as f:
                raw[split] = [(clean(r["text"]), r["category"].strip()) for r in csv.DictReader(f)]

        names = sorted({label for _, label in raw["train"]})
        index = {n: i for i, n in enumerate(names)}
        unknown = {label for rows in raw.values() for _, label in rows} - set(index)
        if unknown:
            raise ValueError(f"labels in the test split that are not in train: {sorted(unknown)}")

        splits = {split: [{"text": t, "label": index[l], "label_name": l} for t, l in rows if t]
                  for split, rows in raw.items()}
        return splits, names, shas