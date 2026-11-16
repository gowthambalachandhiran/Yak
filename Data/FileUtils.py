"""Small helpers shared by the dataset classes: downloading, hashing, writing files, simple statistics.

Standard library only.
"""
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request


def log(msg=""):
    print(msg, flush=True)


def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def download(url, dest, retries=3):
    """Download url to dest, written atomically. Returns the SHA-256 of the file."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "yak-dataset-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done, next_mark, sha = 0, 10, hashlib.sha256()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    sha.update(chunk)
                    done += len(chunk)
                    if total >= 5_000_000 and done * 100 // total >= next_mark:
                        log(f"    {os.path.basename(dest)}: {done * 100 // total}% of {total / 1e6:.0f} MB")
                        next_mark += 10
            if done == 0:
                raise IOError("the server returned an empty file")
            if total and done != total:
                raise IOError(f"incomplete download: got {done} of {total} bytes")
            os.replace(tmp, dest)
            return sha.hexdigest()
        except (urllib.error.URLError, OSError, IOError) as e:
            last_error = e
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < retries:
                wait = 2 ** attempt
                log(f"    attempt {attempt} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"could not download {url}: {last_error}")


def fetch(url, raw_dir, filename):
    """Download into raw_dir unless the file is already there. Returns (path, sha256)."""
    dest = os.path.join(raw_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log(f"  using existing {filename}")
        return dest, sha256_file(dest)
    log(f"  downloading {filename}")
    return dest, download(url, dest)


def write_jsonl(path, rows):
    """Write one JSON object per line, atomically. Returns the number of rows written."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def clean(s):
    """Collapse runs of whitespace and strip the ends."""
    return " ".join(str(s).split())


def length_stats(rows, text_of):
    words = sorted(len(text_of(r).split()) for r in rows)
    return {"mean_words": round(statistics.fmean(words), 1),
            "p95_words": words[int(0.95 * (len(words) - 1))], "max_words": words[-1]}


def overlap_with_train(splits, text_of):
    """How many validation/test examples have exactly the same text as a training example."""
    train_texts = {text_of(r) for r in splits["train"]}
    return {name: sum(1 for r in rows if text_of(r) in train_texts)
            for name, rows in splits.items() if name != "train"}