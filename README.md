\# Yak



Yak is an open source reimplementation of JEV, built from the ground up by a small team. The goal is simple: no black boxes. We write the tokenizer ourselves, pretrain a compact transformer ourselves, fine-tune it ourselves, and measure how well it knows what it doesn't know.



The name comes from the yak, an animal that is small enough to look unassuming and tough enough to work at altitude. That is roughly the spirit we are going for: a modest model that holds up under real evaluation.



\## Status



Early stage. We are following a five-week plan (see the roadmap below). This README will be updated as each stage lands, so some sections describe what we intend to build rather than what exists today.



\## What we are building



Yak is a small encoder-style transformer trained with masked language modeling (MLM), then fine-tuned on a mix of general NLP tasks and evaluated for both accuracy and calibration.



The pipeline has four pieces:



1\. A byte pair encoding (BPE) tokenizer implemented from scratch, with no tokenizer libraries doing the heavy lifting.

2\. A pretraining corpus built from Wikipedia and BookCorpus.

3\. An MLM pretraining run using our own tokenizer.

4\. Fine-tuning with RLCD, followed by evaluation with Brier score and standard benchmarks.



\## Model configuration



| Parameter | Value |

|-----------|-------|

| d\_model | 256 |

| n\_head | 4 |

| n\_layers | 4 |

| d\_ff | 1024 |

| vocab\_size | 30,000 |

| max\_seq\_len | 3,000 |



With 4 heads and a model width of 256, each attention head works in 64 dimensions. Excluding embeddings, the four transformer layers come to roughly 3.1M parameters. The token embedding table adds about 7.7M more (30,000 x 256), so expect a model in the neighborhood of 11M parameters in total, depending on how positional embeddings and weight tying are handled.



This is deliberately small. It lets the whole team train and debug on modest hardware, and it makes every experiment cheap enough to repeat.



\## Roadmap



\### Week 1: BPE from scratch



Implement byte pair encoding ourselves: learn merges from raw text, build the merge table and vocabulary, and write encode and decode functions. The focus is on correctness first (round-trip encoding must be lossless) and speed second.



Deliverable: a standalone BPE module with unit tests.



\### Week 2: Corpus and tokenizer



Assemble the pretraining corpus from Wikipedia and BookCorpus, about 1.5 GB of text in total. Clean and deduplicate it, then train the BPE tokenizer on it with a vocabulary size of 30,000. Sequences will be packed to a context length of 3,000 tokens.



Deliverable: the cleaned corpus pipeline, a trained tokenizer, and a short report on token statistics (average tokens per document, coverage, rare token behavior).



\### Week 3: MLM pretraining



Pretrain the model with a masked language modeling objective using the configuration above and the tokenizer from Week 2. We will track training and validation loss, masked token accuracy, and checkpoint regularly.



Deliverable: a pretrained Yak checkpoint and the training logs.



\### Week 4: Fine-tuning with RLCD



Fine-tune the pretrained model on a mix of general NLP datasets:



\- Banking77 (intent classification)

\- CLINC (intent classification, including out-of-scope queries)

\- SNLI (natural language inference)

\- Yelp (sentiment)



Fine-tuning uses RLCD. Dataset mixing ratios and RLCD hyperparameters will be documented here once we settle them.



Deliverable: a fine-tuned Yak checkpoint and the full training configuration.



\### Week 5: Evaluation



Evaluate Yak on two fronts. First, calibration, measured with the Brier score, which tells us whether the model's confidence matches how often it is actually right. Second, task performance on standard benchmarks for the datasets above.



Deliverable: an evaluation report with Brier scores and benchmark results, plus the scripts to reproduce them.



\## Planned repository layout



This may shift as we build, but the rough shape is:



```

yak/

&#x20; tokenizer/     BPE implementation and tokenizer training

&#x20; data/          corpus download, cleaning, and packing scripts

&#x20; model/         transformer definition and config

&#x20; pretrain/      MLM pretraining loop

&#x20; finetune/      RLCD fine-tuning and dataset mixing

&#x20; eval/          Brier score and benchmark evaluation

&#x20; tests/

&#x20; README.md

```



\## Getting started



Setup instructions will go here once the first modules are in place. The plan is for everything to be runnable from a clean checkout with a single environment file and a handful of documented commands.



\## Contributing



Yak is a team project and we welcome outside help once the core pipeline is stable. If you want to contribute:



\- Open an issue before starting large changes so we can avoid duplicated work.

\- Keep pull requests focused. One change, one PR.

\- Add tests for anything that touches the tokenizer, since silent tokenization bugs are painful to find later.



\## Team



Maintained by the Yak team. Contributor names will be added here.



\## License



To be decided before the first public release.



\## Acknowledgments



Yak is inspired by JEV, and builds on public datasets including Wikipedia, BookCorpus, Banking77, CLINC, SNLI, and Yelp. Thanks to everyone who made those resources available.

