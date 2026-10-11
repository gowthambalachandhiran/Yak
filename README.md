# Yak

Yak is an open source reimplementation of JEV, built from the ground up by a small team. The goal is simple: no black boxes. We write the tokenizer ourselves, pretrain a compact transformer ourselves, fine-tune it ourselves, and measure how well it knows what it doesn't know.

The name comes from the yak, an animal that is small enough to look unassuming and tough enough to work at altitude. That is roughly the spirit we are going for: a modest model that holds up under real evaluation.

## Status

Early stage. The tokenizer (week 1) is done. We are now in week 2, MLM pretraining. The roadmap below shows what each stage covers, and this README will be updated as each stage lands, so some sections describe what we intend to build rather than what exists today.

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
| vocab_size | 30,000 plus 5 special tokens |
| max_seq_len | 2,048 |
| positional encoding | static sinusoidal (sin and cos), not learned |
| special tokens | [PAD], [UNK], [CLS], [SEP], [MASK] |

With 4 heads and a model width of 256, each attention head works in 64 dimensions. Excluding embeddings, the four transformer layers come to roughly 3.1M parameters. The token embedding table adds about 7.7M more (30,000 x 256). Positions use a fixed sin and cos table, so they add no parameters, and the output layer is tied to the input embedding. That puts the whole model at roughly 11M parameters.

This is deliberately small. It lets the whole team train and debug on modest hardware, and it makes every experiment cheap enough to repeat.

## Roadmap

| Week | Stage | What we do | Deliverable |
|------|-------|------------|-------------|
| 1 | Tokenizer (done) | Implement byte pair encoding from scratch: learn merges from raw text, build the merge table and vocabulary, and write encode and decode functions. Train it on the Wikipedia and BookCorpus text with a 30,000 vocabulary, plus special tokens for padding, unknown, classification, separation, and masking. | Standalone BPE module, trained tokenizer file, and token statistics |
| 2 | MLM pretraining | Pretrain the 4 layer model with a masked language modeling objective using our own tokenizer, on sequences packed to a 2,048 token context. Track training and validation loss and masked token accuracy, and checkpoint regularly. Details below. | Pretrained Yak checkpoint and training logs |
| 3 | Fine-tuning with RLCD | Fine-tune on a mix of Banking77, CLINC, SNLI, and Yelp using RLCD. Mixing ratios and hyperparameters will be documented once settled. | Fine-tuned Yak checkpoint and full training configuration |
| 4 | Evaluation | Measure calibration with the Brier score and task performance on standard benchmarks for the datasets above. | Evaluation report and reproduction scripts |

## Week 2: MLM pretraining

This is what we plan to cover this week.

**Data preparation.** Tokenize the full corpus once with our tokenizer and store the ids on disk, so training never tokenizes on the fly. Documents are packed into 2,048 token chunks with a separator token between them and a classification token at the start of each chunk, so almost no compute is wasted on padding. A small validation split is held out by document before packing.

**Model.** An encoder-only transformer using the configuration above: static sinusoidal positions, pre-layer-norm blocks, padding-aware attention, and an output layer tied to the token embedding. We also check that the parameter count matches what we expect.

**Masking.** Masking is dynamic, meaning positions are chosen fresh every time a batch is loaded. We select 15 percent of tokens; of those, 80 percent become the mask token, 10 percent become a random token, and 10 percent stay unchanged. Special tokens are never selected, and loss is computed only on the selected positions.

**Training.** AdamW with a warmup followed by learning rate decay, mixed precision, gradient clipping, and batches sized in tokens rather than sequences. Most steps may run at a shorter sequence length, with a final phase at the full 2,048 context, since the positional table already covers it.

**Monitoring.** We track training loss, validation loss, masked token accuracy, and throughput. Accuracy is compared against a baseline that always predicts the most frequent token, so we know the model is learning more than word frequency. We also run fill-the-mask checks on a fixed set of sentences at each checkpoint.

**Checkpointing.** Checkpoints are saved regularly with optimizer state so a run can be resumed exactly, and the best checkpoint by validation loss is kept.

**Sanity checks before the long run.** The starting loss should be close to the log of the vocabulary size. The model should be able to overfit a single small batch. The measured masking rate should land near 15 percent.

**Done when:**
- Validation loss has flattened or is improving very slowly.
- Masked token accuracy is clearly above the frequency baseline.
- Fill-the-mask outputs look sensible.
- A reloaded checkpoint reproduces the same validation loss.
- The final checkpoint, config, tokenizer file, and logs are saved together so the fine-tuning stage can load them without guesswork.

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