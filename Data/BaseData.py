"""Base class for a dataset that is downloaded and converted to the Yak .jsonl format.

A dataset class only has to say where its files come from and how to turn them into splits of examples
(`build`). Writing the files, the label file, the sanity checks and the statistics are shared here.

Output, under <out_dir>/<name>/:
    <split>.jsonl      one example per line, with "label" (int) and "label_name" plus the text field(s)
    labels.json        {"names": [...], "num_labels": N, ...}  (names are listed in label-id order)
"""
import os

try:                                    # imported as part of the Data package (from Data.Banking77 import ...)
    from .FileUtils import write_jsonl, write_json, length_stats, overlap_with_train
except ImportError:                     # run as a script from inside the Data folder
    from FileUtils import write_jsonl, write_json, length_stats, overlap_with_train


class BaseDataset:
    name = ""               # folder name under the output directory
    expected_splits = {}    # published split sizes: a mismatch prints a warning, it does not stop the run
    expected_labels = 0     # published number of labels
    sources = {}            # where the files come from; recorded in manifest.json

    def __init__(self, raw_dir, out_dir):
        self.raw_dir = raw_dir  # downloaded source files are kept here
        self.out_dir = out_dir  # the dataset's own folder is created inside this one

    # ------------------------------------------------------------------ to be provided by each dataset
    def build(self):
        """Download the source files and return (splits, label_names, raw_sha256).

        splits:       {"train": [row, ...], "validation": [...], "test": [...]}, each row a dict
        label_names:  list of names; the position in the list is the label id
        raw_sha256:   {raw file name: sha256 hex digest}
        """
        raise NotImplementedError

    def text_of(self, row):
        """The text of one example as a single string (used for length statistics and train/test overlap)."""
        raise NotImplementedError

    def extra_label_info(self, names):
        """Optional extra fields for labels.json."""
        return {}

    # ------------------------------------------------------------------ shared
    @property
    def dataset_dir(self):
        return os.path.join(self.out_dir, self.name)

    def is_prepared(self):
        need = [os.path.join(self.dataset_dir, "labels.json")]
        need += [os.path.join(self.dataset_dir, f"{s}.jsonl") for s in self.expected_splits]
        return all(os.path.exists(p) for p in need)

    def prepare(self):
        """Build the dataset, write all files, and return a summary for manifest.json."""
        splits, names, shas = self.build()

        sizes = {split: write_jsonl(os.path.join(self.dataset_dir, f"{split}.jsonl"), rows)
                 for split, rows in splits.items()}
        labels = {"names": names, "num_labels": len(names)}
        labels.update(self.extra_label_info(names))
        write_json(os.path.join(self.dataset_dir, "labels.json"), labels)

        warnings = []
        if sizes != self.expected_splits:
            warnings.append(f"split sizes {sizes} differ from the published {self.expected_splits}")
        if len(names) != self.expected_labels:
            warnings.append(f"{len(names)} labels, expected {self.expected_labels}")
        seen = {r["label"] for rows in splits.values() for r in rows}
        if seen != set(range(len(names))):
            warnings.append(f"labels never used: {sorted(set(range(len(names))) - seen)[:10]}")

        return {
            "sources": self.sources,
            "raw_sha256": shas,
            "splits": sizes,
            "num_labels": len(names),
            "text_length": length_stats(splits["train"], self.text_of),
            "eval_texts_also_in_train": overlap_with_train(splits, self.text_of),
            "warnings": warnings,
        }