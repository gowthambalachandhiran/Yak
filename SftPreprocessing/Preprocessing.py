
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
    """Read CSV, JSONL, or JSON data."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # Support common dataset wrappers.
            for key in ("data", "examples", "records"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])

            return pd.DataFrame(data)

    raise ValueError(f"Unsupported file format: {path}")


def find_column(df, candidates, required=True):
    """Find a column using common names."""
    columns = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_label(value, dataset):
    """Convert labels to consistent strings."""
    if pd.isna(value):
        return None

    label = str(value).strip()

    if not label or label.lower() in {
        "nan", "none", "null", "-",
    }:
        return None

    if dataset == "snli":
        return SNLI_LABEL_MAP.get(label, label.lower())

    return label


def load_dataset_split(dataset, split, relative_path):
    path = ROOT / relative_path
    df = read_file(path)

    text_col = find_column(
        df,
        [
            "text", "utterance", "query", "sentence",
            "input", "premise",
        ],
        required=False,
    )

    label_col = find_column(
        df,
        [
            "label", "labels", "intent", "category",
            "class", "gold_label",
        ],
        required=False,
    )

    records = []

    if dataset == "snli":
        premise_col = find_column(
            df, ["premise", "sentence1"]
        )
        hypothesis_col = find_column(
            df, ["hypothesis", "sentence2"]
        )

        if label_col is None:
            raise ValueError(
                f"No label column found in {path}. "
                f"Available columns: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            label = normalize_label(row[label_col], dataset)

            if label not in {
                "entailment", "neutral", "contradiction"
            }:
                continue

            premise = str(row[premise_col]).strip()
            hypothesis = str(row[hypothesis_col]).strip()

            if not premise or not hypothesis:
                continue

            records.append({
                "source_dataset": dataset,
                "task": DATASETS[dataset]["task"],
                "split": split,
                "text": (
                    f"Premise: {premise}\n"
                    f"Hypothesis: {hypothesis}"
                ),
                "label": label,
            })

    else:
        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not identify text and label columns in "
                f"{path}. Available columns: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            text_value = row[text_col]
            label = normalize_label(row[label_col], dataset)

            if pd.isna(text_value) or label is None:
                continue

            text_value = str(text_value).strip()

            if not text_value:
                continue

            records.append({
                "source_dataset": dataset,
                "task": DATASETS[dataset]["task"],
                "split": split,
                "text": text_value,
                "label": label,
            })

    print(
        f"Loaded {len(records):,} records "
        f"from {dataset}/{split}"
    )

    return records


def load_all_splits():
    """Load configured splits for all three datasets."""
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
    # Reserve 10% of its training records for validation.
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

    balanced = []
    rng = random.Random(SEED)

    for dataset in sorted(grouped):
        rows = grouped[dataset]

        if len(rows) < quota:
            raise ValueError(
                f"{dataset} has {len(rows)} training rows, "
                f"but the requested quota is {quota}."
            )

        rng.shuffle(rows)
        balanced.extend(rows[:quota])

    rng.shuffle(balanced)

    print("\nBalanced training counts:")
    for dataset in sorted(grouped):
        print(f"  {dataset}: {quota:,}")

    return balanced


def convert_to_sft_format(records, label_vocabularies):
    """Convert examples into prompt/target records."""
    converted = []

    for record in records:
        dataset = record["source_dataset"]
        labels = label_vocabularies[dataset].copy()

        # Shuffle candidate labels for training only.
        if record["split"] == "train":
            random.shuffle(labels)

        if dataset == "snli":
            prompt = (
                "Determine the relationship between the premise "
                "and hypothesis.\n"
                "Choose exactly one label from the candidates.\n\n"
                f"{record['text']}\n\n"
                f"Candidate labels: {', '.join(labels)}"
            )
        else:
            prompt = (
                "Classify the user's message into exactly one "
                "of the candidate intent labels.\n\n"
                f"Message: {record['text']}\n\n"
                f"Candidate labels: {', '.join(labels)}"
            )

        converted.append({
            "source_dataset": dataset,
            "task": record["task"],
            "split": record["split"],
            "input": prompt,
            "target": record["label"],
            "candidate_labels": labels,
        })

    return converted


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Saved {len(records):,} records to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced SFT data for Yak."
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help=(
            "Number of training examples per dataset. "
            "Default: size of the smallest training dataset."
        ),
    )
    args = parser.parse_args()

    random.seed(SEED)

    splits = load_all_splits()

    if not splits["train"]:
        raise ValueError("No training data found.")

    # Build one label vocabulary per dataset from training data.
    label_vocabularies = {}

    for dataset in DATASETS:
        labels = sorted({
            r["label"]
            for r in splits["train"]
            if r["source_dataset"] == dataset
        })

        if not labels:
            raise ValueError(
                f"No valid training labels found for {dataset}."
            )

        label_vocabularies[dataset] = labels

    # Balance training only. Validation and test remain unbalanced.
    splits["train"] = balance_training_examples(
        splits["train"],
        samples_per_task=args.samples_per_task,
    )

    converted = {
        split: convert_to_sft_format(
            records, label_vocabularies
        )
        for split, records in splits.items()
    }

    for split, records in converted.items():
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
            for split, records in converted.items()
        },
        "train_counts_by_dataset": {
            dataset: sum(
                r["source_dataset"] == dataset
                for r in converted["train"]
            )
            for dataset in DATASETS
        },
    }

    with (
        OUTPUT_DIR / "label_metadata.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nPreprocessing complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Maximum number of labels: {metadata['max_num_labels']}")


if __name__ == "__main__":
    main()
