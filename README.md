# Yak

Yak is an open source reimplementation of JEV — a simple model from scratch — built from the ground up by a small team. The goal is to build a simple clone from scratch. We write the tokenizer ourselves, pretrain a compact transformer ourselves, fine-tune it ourselves, and measure how well it performs.

## What we are building

Yak is a small transformer trained with masked language modeling (MLM), then fine-tuned on a mix of general NLP tasks and finally adjusted with reinforcement learning.

The pipeline has four pieces:

1. A byte pair encoding (BPE) tokenizer implemented from scratch, with no tokenizer libraries doing the heavy lifting.
2. A pretraining corpus built from Wikipedia and BookCorpus.
3. An MLM pretraining run using our own tokenizer.
4. Fine-tuning with RLCD, followed by evaluation with Brier score and standard benchmarks.

## Model configuration

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_head | 4 |
| n_layers | 4 |
| d_ff | 1024 |
| vocab_size | 30,000 |
| max_seq_len | 3,000 |

With 4 heads and a model width of 256, each attention head works in 64 dimensions. Excluding embeddings, the four transformer layers come to roughly 3.1M parameters. The token embedding table adds about 7.7M more (30,000 x 256), so expect a model in the neighborhood of 11M parameters in total, depending on how positional embeddings and weight tying are handled.

This is a small model for learning. It lets the whole team train and debug on modest hardware, and it makes every experiment cheap enough to repeat.

## Roadmap

| Week | Stage | What we do | Deliverable |
|------|-------|------------|-------------|
| 1 | BPE from scratch | Implement byte pair encoding ourselves: learn merges from raw text, build the merge table and vocabulary, and write encode and decode functions. Round-trip encoding must be lossless. | Standalone BPE module with unit tests |
| 2 | Corpus and tokenizer | Assemble about 1.5 GB of Wikipedia and BookCorpus text, clean and deduplicate it, and train the BPE tokenizer with a 30,000 vocabulary. Sequences are packed to a 3,000 token context. | Corpus pipeline, trained tokenizer, and a token statistics report |
| 3 | MLM pretraining | Pretrain the 4 layer model with a masked language modeling objective using our own tokenizer. Track training and validation loss and masked token accuracy, and checkpoint regularly. | Pretrained Yak checkpoint and training logs |
| 4 | Fine-tuning with RLCD | Fine-tune on a mix of Banking77, CLINC, SNLI, and Yelp using RLCD. Mixing ratios and hyperparameters will be documented once settled. | Fine-tuned Yak checkpoint and full training configuration |
| 5 | Evaluation | Measure calibration with the Brier score and task performance on standard benchmarks for the datasets above. | Evaluation report and reproduction scripts |

## Planned repository layout

```
yak/
  __init__.py    Package root
  tokeniser/     BPE implementation and tokenizer training
  data/          corpus download, cleaning, and packing scripts (planned)
  model/         transformer definition and config (planned)
  pretrain/      MLM pretraining loop (planned)
  finetune/      RLCD fine-tuning and dataset mixing (planned)
  eval/          Brier score and benchmark evaluation (planned)
tests/
  conftest.py
  tokeniser/
    test_bpe.py
```

## Getting started

Project Yak is maintained using [uv](https://docs.astral.sh/uv/).

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/#installation)

### Install dependencies

```bash
# Install all dependencies (including dev dependencies)
uv sync
```

### Run tests

```bash
uv run pytest
```

## Acknowledgments

Yak is inspired by JEV, and builds on public datasets including Wikipedia, BookCorpus, Banking77, CLINC, SNLI, and Yelp. Thanks to everyone who made those resources available.
