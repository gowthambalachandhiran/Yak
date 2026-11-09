#!/usr/bin/env python3
"""Download and prepare the Yak fine-tuning datasets: Banking77, CLINC (150 intents + out-of-scope) and SNLI.

This is the one file to run. The work is done by the classes next to it:
    FileUtils.py     download, hashing and file-writing helpers
    BaseDataset.py   shared logic: writing files, label file, sanity checks, statistics
    Banking77.py     class Banking77
    Clinc.py         class Clinc
    Snli.py          class Snli

Usage (from anywhere):
    python Data/download_datasets.py                      # all three datasets
    python Data/download_datasets.py --only banking77 clinc
    python Data/download_datasets.py --force              # redo everything
    python Data/download_datasets.py --out /some/folder   # default: the folder this script lives in

Output, under --out:
    _raw/                         downloaded source files (kept so a re-run works offline; --delete-raw removes them)
    banking77/{train,test}.jsonl
    clinc/{train,validation,test}.jsonl
    snli/{train,validation,test}.jsonl
    <dataset>/labels.json         label names in id order
    manifest.json                 sources, SHA-256 of every raw file, split sizes, label counts, largest label space

To add a dataset: write a class like Banking77 (subclass BaseDataset, implement build and text_of) and add it to DATASETS.
"""
import argparse
import datetime
import json
import os
import sys
import ssl
import urllib.request

# Bypass SSL verification locally (useful if behind a corporate proxy / Zscaler / SSL inspection)
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # lets the class files import each other when run as a script

from FileUtils import log, write_json
from Banking77 import Banking77
from Clinic import Clinc
from Snli import Snli, DEFAULT_URL as SNLI_URL

DATASETS = {"banking77": Banking77, "clinc": Clinc, "snli": Snli}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Download and prepare Banking77, CLINC and SNLI.")
    ap.add_argument("--out", default=here, help="output folder (default: the folder this script is in)")
    ap.add_argument("--only", nargs="+", choices=sorted(DATASETS), help="only these datasets")
    ap.add_argument("--force", action="store_true", help="rebuild even if the outputs already exist")
    ap.add_argument("--delete-raw", action="store_true", help="delete the downloaded source files when finished")
    ap.add_argument("--snli-url", default=SNLI_URL, help="alternative location of snli_1.0.zip (a mirror or a local file:// URL)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    raw_dir = os.path.join(out_dir, "_raw")
    os.makedirs(out_dir, exist_ok=True)
    #Will update the code

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = {"datasets": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.setdefault("datasets", {})

    failed = []
    for name in (args.only or list(DATASETS)):
        log(f"[{name}]")
        dataset = Snli(raw_dir, out_dir, url=args.snli_url) if name == "snli" else DATASETS[name](raw_dir, out_dir)
        if not args.force and name in manifest["datasets"] and dataset.is_prepared():
            log("  already prepared (use --force to redo)")
            continue
        try:
            summary = dataset.prepare()
            summary["prepared_at_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            manifest["datasets"][name] = summary
            for w in summary["warnings"]:
                log(f"  WARNING: {w}")
        except Exception as e:                                # keep going so one broken source does not block the others
            failed.append(name)
            manifest["datasets"].pop(name, None)
            log(f"  FAILED: {e}")
        counts = [d["num_labels"] for d in manifest["datasets"].values()]
        manifest["max_num_labels"] = max(counts) if counts else None
        write_json(manifest_path, manifest)

    if args.delete_raw and not failed and os.path.isdir(raw_dir):
        for fn in os.listdir(raw_dir):
            os.remove(os.path.join(raw_dir, fn))
        os.rmdir(raw_dir)

    log("\nSummary")
    log(f"{'dataset':<11}{'labels':>7}  {'splits':<46}{'words/ex (mean, p95, max)':<28}eval texts also in train")
    for name, d in manifest["datasets"].items():
        splits = ", ".join(f"{k} {v:,}" for k, v in d["splits"].items())
        t = d["text_length"]
        log(f"{name:<11}{d['num_labels']:>7}  {splits:<46}{str((t['mean_words'], t['p95_words'], t['max_words'])):<28}{d['eval_texts_also_in_train']}")
    if manifest.get("max_num_labels"):
        biggest = max(manifest["datasets"], key=lambda n: manifest["datasets"][n]["num_labels"])
        log(f"\nLargest label space: {biggest} with {manifest['max_num_labels']} labels -> use max_label = {manifest['max_num_labels']}")
    log(f"Output folder: {out_dir}")
    if failed:
        log(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()