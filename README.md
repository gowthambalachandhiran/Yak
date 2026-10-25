# Yak

Yak is an open source reimplementation of JEV, built from the ground up by a small team. The goal is simple: no black boxes. We write the tokenizer ourselves, pretrain a compact transformer ourselves, fine-tune it ourselves, and measure how well it knows what it doesn't know.

The name comes from the yak, an animal that is small enough to look unassuming and tough enough to work at altitude. That is roughly the spirit we are going for: a modest model that holds up under real evaluation.

## Status

| Stage | State |
|-------|-------|
| Tokenizer | Done |
| MLM pretraining | Done. A pretrained checkpoint exists (results below). |
| Dataset download for fine-tuning | Done. One script fetches all three datasets. |
| Supervised fine-tuning | Next. The plan is below. |
| Evaluation (accuracy, Brier score, calibration) | After fine-tuning |
| RLCD (Reinforcement Learning for Calibrated Decisions) with a Brier score reward | After supervised fine-tuning. The plan is below. |

This README is updated as each stage lands. Sections marked "planned" describe what we intend to build, not what exists today.

## What we are building

Yak is a small encoder-style transformer trained with masked language modeling (MLM), then fine-tuned for classification on general NLP tasks and evaluated for both accuracy and calibration.

The pipeline has six pieces:

1. A byte pair encoding (BPE) tokenizer implemented from scratch, with no tokenizer libraries doing the heavy lifting.
2. A pretraining corpus. We used WikiText-103, a cleaned Wikipedia dump. Adding BookCorpus is still possible later.
3. An MLM pretraining run using our own tokenizer.
4. Supervised fine-tuning for classification on Banking77, CLINC and SNLI.
5. RLCD, which trains the fine-tuned model to give calibrated probabilities, using the Brier score as the reward.
6. Evaluation with accuracy, Brier score and calibration measures.

## Model configuration

This is the configuration of the pretrained checkpoint.

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_head | 4 |
| n_layers | 4 |
| d_ff | 512 |
| vocab_size | 30,005 (30,000 vocabulary entries plus 5 special tokens) |
| max_seq_len | 2,048 |
| positional encoding | static sinusoidal (sin and cos), not learned |
| special tokens | [PAD], [UNK], [CLS], [SEP], [MASK] |

With 4 heads and a model width of 128, each attention head works in 32 dimensions. The model has about 4.7M parameters. Of those, 3.84M are the token embedding table, which is shared with the output layer, and only 0.79M are in the four transformer layers. Positions use a fixed sin and cos table, so they add no parameters.

The embedding table is most of the model. That is the main reason it is capacity limited: during pretraining, training loss and validation loss ended up equal, so more steps help but with shrinking returns.

We also ran a larger variant (d_model 256, 6 layers, about 12.4M parameters) as an experiment. It was only trained for a short run, so its numbers are not comparable with the checkpoint above. The notebook is in `Model/Yak_MLM_12M.ipynb`.

## Pretraining results

Pretraining ran on WikiText-103 in Colab and Kaggle notebooks, in several phases with a warm restart of the learning rate at the start of each. Batches were 131,072 tokens (64 sequences of 2,048 tokens). Dropout was turned off in the later phases. About 0.5 percent of articles are held out for validation.

| Checkpoint | Validation loss | Masked token accuracy | Lift over the baseline |
|------------|-----------------|-----------------------|------------------------|
| Frequency baseline (always predict "the") | n/a | 4.77% | n/a |
| Step 24,000 | 4.34 | 33.1% | +28.3 points, 6.9x |
| Step 36,000 | 3.40 | 43.4% | +38.6 points, 9.1x |
| Latest checkpoint | n/a | about 48% | +43.2 points, 10.1x |

Accuracy is measured on masked positions only, with a fixed set of masks so numbers stay comparable between runs. The baseline is the accuracy of a model that ignores the context and always guesses the most frequent token, which is "the", and it scores 4.77%. Lift is how far a checkpoint sits above that baseline, in percentage points and as a multiple. From the 4.77% baseline to about 48% is a lift of roughly 43 points, about ten times the baseline.

Checkpoints are not stored in the repository because of their size. They are kept in cloud storage and are excluded by `.gitignore`.

## Roadmap

| Week | Stage | What we do | Deliverable |
|------|-------|------------|-------------|
| 1 | Tokenizer (done) | Implement byte pair encoding from scratch, train it with a 30,000 vocabulary, and add special tokens for padding, unknown, classification, separation and masking. | Standalone BPE module, trained tokenizer file (`Tokenizer/TokenFile/bpe.json`), and token statistics |
| 2 | MLM pretraining (done) | Pretrain the 4 layer model with a masked language modeling objective on sequences packed to a 2,048 token context. | Pretrained Yak checkpoint and training logs |
| 3 | Supervised fine-tuning (next) | Download Banking77, CLINC and SNLI, adapt the text to the tokenizer, and fine-tune the pretrained encoder for classification. Details below. | Fine-tuned checkpoints, training configuration, and baselines |
| 4 | RLCD (planned) | Train the fine-tuned models with Reinforcement Learning for Calibrated Decisions, using the Brier score as the reward, so that their probabilities can be trusted. Details below. | Calibrated checkpoints and training configuration |
| 5 | Evaluation | Measure accuracy, Brier score and calibration on each task, and compare against baselines. | Evaluation report and reproduction scripts |

## Week 3: Supervised fine-tuning (planned)

**Datasets.** Three classification datasets, downloaded by one script (see Getting started).

| Dataset | Task | Labels | Splits |
|---------|------|--------|--------|
| Banking77 | Banking intents | 77 | train 10,003, test 3,080 (no validation split) |
| CLINC | 150 intents plus out-of-scope | 151 | train 15,100, validation 3,100, test 5,500 |
| SNLI | Sentence pair inference | 3 | train 549,367, validation 9,842, test 9,824 (published sizes) |

The largest label space is CLINC with 151 labels, so the output layer has `max_label = 151` outputs. For CLINC the out-of-scope class is the last label, id 150. Banking77 has no validation split, so we create one from its training set with a fixed seed.

**Matching the text to the tokenizer.** The tokenizer was trained on WikiText, where punctuation is separated from words by spaces and text keeps its case. Downstream text such as `card?` would otherwise be split into odd pieces or `[UNK]`. We apply the same spacing before tokenizing and measure the `[UNK]` rate on each dataset before training anything.

**Inputs.** A single text becomes `[CLS] text [SEP]`. An SNLI pair becomes `[CLS] premise [SEP] hypothesis [SEP]`. Inputs are truncated (about 128 tokens is enough for these datasets) and padded dynamically. The model has no segment embeddings, so SNLI is expected to be the hardest task.

**Model.** The pretrained encoder with the masked language modeling head removed, a pooling step, and a linear output layer with 151 outputs. We use mean pooling over real tokens by default, because `[CLS]` was never trained to summarise a sentence during MLM, and we compare it against `[CLS]` pooling. Dropout is switched back on for fine-tuning, because these datasets are small.

**Labels across tasks.** Each task only uses its own number of outputs. The extra outputs are cut off before the softmax, so they take no probability and the Brier score is computed over real classes only. A separate label-mixing module maps each dataset's labels into the shared outputs, so the model can be trained on a mix of tasks. It supports two ways of doing this, which we will compare:

- **Shared rows:** every task uses outputs 0 to K-1 of the same layer.
- **Disjoint rows:** each task gets its own non-overlapping range of outputs, 233 in total for the three datasets plus any future ones.

**Baselines.** Three checks before trusting any fine-tuning result: a linear classifier on the frozen encoder, the same architecture trained from random weights, and a TF-IDF with logistic regression model. We also confirm the model can overfit one small batch.

**Training.** AdamW with warm-up and linear decay, 3 to 5 epochs, batches of 32 to 64, early stopping on validation accuracy, and 3 seeds per setting. Learning rates around 1e-4 to 3e-4 are the starting point, higher than usual for larger models because this one is small. These values will be tuned and then recorded here.

**Done when:**
- Each dataset has a fine-tuned checkpoint that beats the from-scratch and TF-IDF baselines on validation data.
- Results are reported as mean and spread over seeds, on the official test splits.
- The fine-tuning configuration, label mapping and tokenizer file are saved with each checkpoint.
- A mixed-task run has been compared with the single-task runs.

## Week 4: RLCD with a Brier score reward (planned)

RLCD stands for Reinforcement Learning for Calibrated Decisions. The idea is that the model returns a decision together with calibrated probabilities, rather than generated text. It is unrelated to Reinforcement Learning from Contrastive Distillation, which shares the abbreviation.

**Reward.** We start from each task's best supervised checkpoint. For every example the reward is the negative Brier score of the predicted class probabilities against the true label:

```
Brier = sum over classes k of (p_k - y_k)^2        y_k is 1 for the true class, 0 for the others
```

A confident correct prediction scores 0, the best possible value, and a confident wrong one scores 2, the worst. The Brier score is a strictly proper scoring rule, so the model earns the highest expected reward by reporting what it truly believes. It is penalised both for confident mistakes and for being unsure about answers it gets right.

**Plan.**
- Start from the supervised checkpoints, so RLCD only has to improve calibration and does not have to learn the task.
- Probabilities are computed over a task's real classes only, using the same label handling as in fine-tuning.
- The reward is a smooth function of the predicted probabilities. We first optimise it directly, which is equivalent to training on the Brier loss, and add sampling-based reinforcement learning only if we need the model to sample its decisions.
- Track accuracy next to the reward. If accuracy falls noticeably, we stop or lower the learning rate. The tolerance is set before the run.
- Compare against cheap calibration baselines, namely temperature scaling and label smoothing. RLCD has to beat them to justify itself.

**Done when:**
- On each task, the test-set Brier score and expected calibration error are lower than for both the supervised checkpoint and its temperature-scaled version.
- Accuracy stays within the agreed tolerance of the supervised checkpoint.
- Results are reported as mean and spread over 3 seeds.

## Week 5: Evaluation (planned)

For every task we report accuracy and macro F1, plus calibration: Brier score, expected calibration error and a reliability diagram. A temperature fitted on validation data is applied to the logits, and we report calibration before and after. For CLINC we also report out-of-scope handling separately.

Because the Brier score is both the RLCD reward and a headline metric, we also report log loss and expected calibration error, which the training does not target directly. All tuning uses validation data only, and the test splits stay untouched until the final evaluation.

## Repository layout

```
Data/
  DownloadData.py     the one script to run; downloads and prepares the datasets
  BaseDataset.py      shared logic for dataset classes
  Banking77.py        Banking77 dataset class
  Clinc.py            CLINC dataset class
  Snli.py             SNLI dataset class
  FileUtils.py        download and file helpers
Model/                transformer layers, encoder block and the MLM model
Tokenizer/            BPE implementation and the trained tokenizer file
Finetune/             (planned) classification model, label mixing, supervised fine-tuning and RLCD
Eval/                 (planned) Brier score, calibration and benchmark evaluation
README.md
```

Downloaded datasets are written next to the scripts in `Data/` (as `banking77/`, `clinc/`, `snli/`, `_raw/` and `manifest.json`) and are excluded by `.gitignore`. Only the Python files in `Data/` are tracked. Model checkpoints (`*.pt`, `*.ckpt`) and tokenized caches (`*.npy`) are excluded as well.

## Getting started

### Download the fine-tuning datasets

The script uses only the Python standard library. From the repository root:

```
python Data/DownloadData.py
```

On Windows use `python`, not `python3`. Useful options:

```
python Data/DownloadData.py --only banking77 clinc   # a subset
python Data/DownloadData.py --force                  # redo everything
python Data/DownloadData.py --out <folder>           # write somewhere else
python Data/DownloadData.py --delete-raw             # remove the downloaded source files at the end
```

Each dataset is written as `train.jsonl`, `validation.jsonl` (where it exists) and `test.jsonl`, plus a `labels.json`. The script ends by printing the largest label space, which should read `max_label = 151`. A re-run skips datasets that are already prepared.

If the download fails with a certificate error, which can happen behind a company proxy or antivirus that inspects HTTPS, try `pip install --upgrade certifi`, or pass `--cafile` with your organisation's certificate bundle. `--insecure` turns certificate checking off and should be a last resort. If a server cannot be reached, download the file in a browser and save it into `Data/_raw/` under the name shown in the error message. The script then uses it.

### Other setup

Setup instructions for pretraining and fine-tuning will be added as those modules land. The plan is for everything to be runnable from a clean checkout with a single environment file and a handful of documented commands.

## Contributing

Yak is a team project and we welcome outside help once the core pipeline is stable. If you want to contribute:

- Open an issue before starting large changes so we can avoid duplicated work.
- Keep pull requests focused. One change, one PR.
- Add tests for anything that touches the tokenizer, since silent tokenization bugs are painful to find later.
- Do not commit datasets or checkpoints.

## Team

Maintained by the Yak team. Contributor names will be added here.

## License

To be decided before the first public release.

## Acknowledgments

Yak is inspired by JEV, and builds on public datasets including WikiText-103, Banking77, CLINC and SNLI. Thanks to everyone who made those resources available.