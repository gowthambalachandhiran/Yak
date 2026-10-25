"""SNLI: sentence pairs labelled entailment, neutral or contradiction.

Source: snli_1.0.zip from the Stanford NLP site (about 100 MB). It contains snli_1.0_train.jsonl, snli_1.0_dev.jsonl and
snli_1.0_test.jsonl. Pass `url` to use a mirror or a local file:// URL instead.
Splits: train, validation (the "dev" file), test.
Label ids: entailment = 0, neutral = 1, contradiction = 2. Pairs without a gold label (marked "-", where the
annotators did not agree) are dropped.
Each example has "premise" and "hypothesis" instead of "text".
"""
import io
import json
import os
import zipfile

try:
    from .BaseData import BaseDataset
    from .FileUtils import fetch, clean, log
except ImportError:
    from BaseData import BaseDataset
    from FileUtils import fetch, clean, log

DEFAULT_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
LABELS = ["entailment", "neutral", "contradiction"]
MEMBERS = {"train": "snli_1.0_train.jsonl", "validation": "snli_1.0_dev.jsonl", "test": "snli_1.0_test.jsonl"}


class Snli(BaseDataset):
    name = "snli"
    expected_splits = {"train": 549367, "validation": 9842, "test": 9824}
    expected_labels = 3

    def __init__(self, raw_dir, out_dir, url=None):
        super().__init__(raw_dir, out_dir)
        self.url = url or DEFAULT_URL
        self.sources = {"zip": self.url}

    def text_of(self, row):
        return row["premise"] + " " + row["hypothesis"]

    def build(self):
        path, sha = fetch(self.url, self.raw_dir, "snli_1.0.zip")
        label_id = {n: i for i, n in enumerate(LABELS)}
        splits, dropped = {}, {}
        with zipfile.ZipFile(path) as zf:
            by_name = {os.path.basename(n): n for n in zf.namelist()}
            for split, member in MEMBERS.items():
                if member not in by_name:
                    raise ValueError(f"{member} is not inside {os.path.basename(path)}")
                rows, skipped = [], 0
                with zf.open(by_name[member]) as raw:
                    for line in io.TextIOWrapper(raw, encoding="utf-8"):
                        ex = json.loads(line)
                        label = ex.get("gold_label")
                        if label not in label_id:             # "-" means the annotators did not agree
                            skipped += 1
                            continue
                        premise, hypothesis = clean(ex["sentence1"]), clean(ex["sentence2"])
                        if premise and hypothesis:
                            rows.append({"premise": premise, "hypothesis": hypothesis,
                                         "label": label_id[label], "label_name": label})
                splits[split], dropped[split] = rows, skipped
        log(f"  dropped examples without a gold label: {dropped}")
        return splits, list(LABELS), {"snli_1.0.zip": sha}