
import argparse
import json
import random
from pathlib import Path

import pandas as pd


SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "Data" / "processed"

DATASETS = {
    "snli": {
        "task": "natural_language_inference",
        "files": {
            "train": "Data/snli_train.csv",
            "validation": "Data/snli_validation.csv",
            "test": "Data/snli_test.csv",
        },
    },
    "banking77": {
        "task": "intent_classification",
        "files": {
            "train": "Data/banking77/train.jsonl",
            "test": "Data/banking77/test.jsonl",
        },
    },
    "clinc": {
        "task": "intent_classification",
        "files": {
            "train": "Data/clinc/train.jsonl",
            "validation": "Data/clinc/validation.jsonl",
            "test": "Data/clinc/test.jsonl",
        },
    },
}

SNLI_LABEL_MAP = {
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction",
}


def read_file(path):
    """Read a supported dataset file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            for key in ("data", "examples", "records"):
                if isinstance(data.get(key), list):
                    return pd.DataFrame(data[key])

        raise ValueError(f"Unexpected JSON structure: {path}")

    raise ValueError(f"Unsupported file format: {path}")


def find_column(df, candidates, required=True):
    """Find a column case-insensitively."""
    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates} in the file. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_label(value, dataset):
    """Normalize a label without losing its readable name."""
    if pd.isna(value):
        return None

    label = str(value).strip()

    if not label or label.lower() in {
        "nan", "none", "null", "-"
    }:
        return None

    if dataset == "snli":
        return SNLI_LABEL_MAP.get(label, label.lower())

    return label


def load_dataset_split(dataset, split, relative_path):
    """Load examples while preserving their original input fields."""
    path = ROOT / relative_path
    df = read_file(path)

    if dataset == "snli":
        premise_col = find_column(
            df, ["premise", "sentence1"]
        )
        hypothesis_col = find_column(
            df, ["hypothesis", "sentence2"]
        )

        # SNLI uses its gold classification label.
        label_col = find_column(
            df, ["gold_label", "label", "labels"]
        )

    else:
        # Prefer readable names over numeric IDs.
        label_col = find_column(
            df,
            [
                "label_name",
                "intent_name",
                "category_name",
                "label",
                "labels",
                "intent",
                "category",
                "class",
            ],
        )

        text_col = find_column(
            df,
            [
                "text",
                "utterance",
                "query",
                "sentence",
                "input",
            ],
        )

    records = []

    for _, row in df.iterrows():
        label = normalize_label(row[label_col], dataset)

        if label is None:
            continue

        if dataset == "snli":
            if label not in {
                "entailment",
                "neutral",
                "contradiction",
            }:
                continue

            premise = row[premise_col]
            hypothesis = row[hypothesis_col]

            if pd.isna(premise) or pd.isna(hypothesis):
                continue

            premise = str(premise).strip()
            hypothesis = str(hypothesis).strip()

            if not premise or not hypothesis:
                continue

            record = {
                "source_dataset": dataset,
                "task": DATASETS[dataset]["task"],
                "split": split,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
            }

        else:
            text_value = row[text_col]

            if pd.isna(text_value):
                continue

            text_value = str(text_value).strip()

            if not text_value:
                continue

            record = {
                "source_dataset": dataset,
                "task": DATASETS[dataset]["task"],
                "split": split,
                "text": text_value,
                "label": label,
            }

        records.append(record)

    print(
        f"Loaded {len(records):,} records "
        f"from {dataset}/{split}"
    )

    return records


def load_all_splits():
    """Load configured splits and reserve Banking77 validation data."""
    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for dataset, config in DATASETS.items():
        for split, relative_path in config["files"].items():
            records = load_dataset_split(
                dataset, split, relative_path
            )
            splits[split].extend(records)

    # Banking77 has no validation file in the current structure.
    # Reserve 10% of its training data for validation.
    banking_train = [
        r for r in splits["train"]
        if r["source_dataset"] == "banking77"
    ]

    if banking_train:
        rng = random.Random(SEED)
        rng.shuffle(banking_train)

        n_validation = max(
            1, int(len(banking_train) * 0.10)
        )

        validation_records = banking_train[:n_validation]
        remaining_records = banking_train[n_validation:]

        splits["train"] = [
            r for r in splits["train"]
            if r["source_dataset"] != "banking77"
        ] + remaining_records

        for record in validation_records:
            record["split"] = "validation"

        splits["validation"].extend(validation_records)

        print(
            "Created Banking77 validation split: "
            f"{len(validation_records):,} records"
        )

    return splits


def balance_training_examples(records, samples_per_task=None):
    """Sample equal numbers of training examples per dataset."""
    grouped = {}

    for record in records:
        grouped.setdefault(
            record["source_dataset"], []
        ).append(record)

    if not grouped:
        raise ValueError("No training examples were loaded.")

    if samples_per_task is None:
        quota = min(len(rows) for rows in grouped.values())
    else:
        quota = samples_per_task

    if quota < 1:
        raise ValueError("Training quota must be at least 1.")

    rng = random.Random(SEED)
    balanced = []

    for dataset in sorted(grouped):
        rows = grouped[dataset]

        if len(rows) < quota:
            raise ValueError(
                f"{dataset} has {len(rows)} records, "
                f"but the requested quota is {quota}."
            )

        balanced.extend(rng.sample(rows, quota))

    rng.shuffle(balanced)

    print("\nBalanced training counts:")
    for dataset in sorted(grouped):
        print(f"  {dataset}: {quota:,}")

    return balanced


def save_jsonl(records, path):
    """Save normalized records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Saved {len(records):,} records to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw, normalized SFT examples for Yak."
    )

    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help=(
            "Number of training examples per dataset. "
            "Defaults to the size of the smallest training dataset."
        ),
    )

    args = parser.parse_args()

    if args.samples_per_task is not None:
        if args.samples_per_task < 1:
            parser.error("--samples-per-task must be at least 1")

    splits = load_all_splits()

    if not splits["train"]:
        raise ValueError("No training data found.")

    # Build label vocabularies BEFORE balancing so that rare
    # labels are not lost from metadata when training rows are sampled.
    label_vocabularies = {}

    for dataset in DATASETS:
        labels = sorted({
            record["label"]
            for record in splits["train"]
            if record["source_dataset"] == dataset
        })

        if not labels:
            raise ValueError(
                f"No training labels found for {dataset}."
            )

        label_vocabularies[dataset] = labels

    # Balance training only. Keep validation and test unchanged.
    splits["train"] = balance_training_examples(
        splits["train"],
        samples_per_task=args.samples_per_task,
    )

    # No prompt construction, candidate labels, or label shuffling.
    for split, records in splits.items():
        random.Random(SEED).shuffle(records)

        save_jsonl(
            records,
            OUTPUT_DIR / f"{split}.jsonl",
        )

    metadata = {
        "seed": SEED,
        "max_num_labels": max(
            len(labels)
            for labels in label_vocabularies.values()
        ),
        "label_vocabularies": label_vocabularies,
        "label_counts": {
            dataset: len(labels)
            for dataset, labels in label_vocabularies.items()
        },
        "split_counts": {
            split: len(records)
            for split, records in splits.items()
        },
        "train_counts_by_dataset": {
            dataset: sum(
                record["source_dataset"] == dataset
                for record in splits["train"]
            )
            for dataset in DATASETS
        },
    }

    with (
        OUTPUT_DIR / "label_metadata.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nPreprocessing complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        "Maximum number of labels: "
        f"{metadata['max_num_labels']}"
    )


if __name__ == "__main__":
    main()
