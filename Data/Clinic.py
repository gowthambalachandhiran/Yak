"""CLINC150 with out-of-scope examples: 150 intents across 10 domains, plus a catch-all "oos" label.

Source: data_full.json from the clinc/oos-eval repository on GitHub. It holds the in-scope splits (train, val, test)
and the out-of-scope splits (oos_train, oos_val, oos_test); the out-of-scope examples are merged into the matching split.
Splits: train, validation, test.
Label ids: the 150 in-scope intents sorted alphabetically are ids 0-149, and "oos" is always id 150 (151 labels in total).
"""
import json

try:
    from .BaseData import BaseDataset
    from .FileUtils import fetch, clean
except ImportError:
    from BaseData import BaseDataset
    from FileUtils import fetch, clean

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"

# our split name -> the keys in data_full.json that make it up (in-scope first, then out-of-scope)
SPLIT_KEYS = {
    "train": ("train", "oos_train"),
    "validation": ("val", "oos_val"),
    "test": ("test", "oos_test"),
}


class Clinc(BaseDataset):
    name = "clinc"
    expected_splits = {"train": 15100, "validation": 3100, "test": 5500}
    expected_labels = 151
    sources = {"full": URL}

    def text_of(self, row):
        return row["text"]

    def extra_label_info(self, names):
        return {"oos_index": names.index("oos")}

    def build(self):
        path, sha = fetch(URL, self.raw_dir, "clinc_data_full.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        in_scope = sorted({label for _, label in data["train"]})
        if "oos" in in_scope:
            raise ValueError("the in-scope training data unexpectedly contains the 'oos' label")
        names = in_scope + ["oos"]                        # out-of-scope is always the last id
        index = {n: i for i, n in enumerate(names)}

        splits = {}
        for split, keys in SPLIT_KEYS.items():
            rows = []
            for key in keys:
                for text, label in data[key]:
                    text = clean(text)
                    if text:
                        rows.append({"text": text, "label": index[label], "label_name": label})
            splits[split] = rows
        return splits, names, {"clinc_data_full.json": sha}