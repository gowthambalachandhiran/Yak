# Yak

Yak is an open source reimplementation of JEV, built from the ground up by a small team. The goal is simple: no black boxes. We write the tokenizer ourselves, pretrain a compact transformer ourselves, fine-tune it ourselves, and measure how well it knows what it doesn't know.

The name comes from the yak, an animal that is small enough to look unassuming and tough enough to work at altitude. That is roughly the spirit we are going for: a modest model that holds up under real evaluation.

## Status

Early stage. We are following a five-week plan (see the roadmap below). This README will be updated as each stage lands, so some sections describe what we intend to build rather than what exists today.

## What we are building

Yak is a small encoder-style transformer trained with masked language modeling (MLM), then fine-tuned on a mix of general NLP tasks and evaluated for both accuracy and calibration.

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

This is deliberately small. It lets the whole team train and debug on modest hardware, and it makes every experiment cheap enough to repeat.

## Roadmap

| Week | Stage | What we do | Deliverable |
|------|-------|------------|-------------|
| 1 | BPE from scratch | Implement byte pair encoding ourselves: learn merges from raw text, build the merge table and vocabulary, and write encode and decode functions. Round-trip encoding must be lossless. | Standalone BPE module with unit tests |
| 2 | Corpus and tokenizer | Assemble about 1.5 GB of Wikipedia and BookCorpus text, clean and deduplicate it, and train the BPE tokenizer with a 30,000 vocabulary. Sequences are packed to a 3,000 token context. | Corpus pipeline, trained tokenizer, and a token statistics report |
| 3 | MLM pretraining | Pretrain the 4 layer model with a masked language modeling objective using our own tokenizer. Track training and validation loss and masked token accuracy, and checkpoint regularly. | Pretrained Yak checkpoint and training logs |
| 4 | Fine-tuning with RLCD | Fine-tune on a mix of Banking77, CLINC, SNLI, and Yelp using RLCD. Mixing ratios and hyperparameters will be documented once settled. | Fine-tuned Yak checkpoint and full training configuration |
| 5 | Evaluation | Measure calibration with the Brier score and task performance on standard benchmarks for the datasets above. | Evaluation report and reproduction scripts |

## Planned repository layout

This may shift as we build, but the rough shape is:

```
yak/
  tokenizer/     BPE implementation and tokenizer training
  data/          corpus download, cleaning, and packing scripts
  model/         transformer definition and config
  pretrain/      MLM pretraining loop
  finetune/      RLCD fine-tuning and dataset mixing
  eval/          Brier score and benchmark evaluation
  tests/
  README.md
```

## Getting started

Setup instructions will go here once the first modules are in place. The plan is for everything to be runnable from a clean checkout with a single environment file and a handful of documented commands.

## Contributing

Yak is a team project and we welcome outside help once the core pipeline is stable. If you want to contribute:

- Open an issue before starting large changes so we can avoid duplicated work.
- Keep pull requests focused. One change, one PR.
- Add tests for anything that touches the tokenizer, since silent tokenization bugs are painful to find later.

## Team

Maintained by the Yak team. Contributor names will be added here.

## License

To be decided before the first public release.

## Acknowledgments

Yak is inspired by JEV, and builds on public datasets including Wikipedia, BookCorpus, Banking77, CLINC, SNLI, and Yelp. Thanks to everyone who made those resources available.