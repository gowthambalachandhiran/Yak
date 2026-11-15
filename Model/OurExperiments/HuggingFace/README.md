---
library_name: transformers
tags:
- yak
- candidate-label-classification
- banking77
- clinc
- snli
---

# Yak Candidate Classifier

Yak is a candidate-label text classifier. At inference time, provide an input and
the labels to compare. The model appends each candidate label to the encoded input,
scores the first token of each label, and returns normalized scores over the
provided candidates.

This repository contains the model weights, configuration, Yak BPE tokenizer data,
label metadata, and the standalone `yak_inference.py` runtime.

## Inference in Google Colab

Start a new Colab notebook and run:

```python
!pip -q install huggingface_hub transformers safetensors numpy

import sys
import torch
from huggingface_hub import snapshot_download

repo_id = "gbalachandhiran/yak-candidate-classifier"
model_dir = snapshot_download(repo_id=repo_id)
sys.path.insert(0, model_dir)

from yak_inference import YakClassifier

classifier = YakClassifier.from_pretrained(
    model_dir,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

candidate_labels = [
    "transfer money",
    "check account balance",
    "report a lost card",
    "cancel a transaction",
]
results = classifier.predict(
    "I need to transfer money to another account",
    candidate_labels=candidate_labels,
)

for result in results:
    print(f"{result['label']}: {result['score']:.4f}")
```

For SNLI-style sentence-pair inference, call `predict` with `premise` and
`hypothesis` instead of a text string:

```python
results = classifier.predict(
    candidate_labels=["entailment", "neutral", "contradiction"],
    premise="A person is riding a bicycle.",
    hypothesis="Someone is cycling.",
)
print(results)
```

## Inference in a local Python environment

Install the requirements:

```bash
pip install -r requirements.txt
```

Download the repository and load the packaged helper:

```python
import sys
from huggingface_hub import snapshot_download

model_dir = snapshot_download("gbalachandhiran/yak-candidate-classifier")
sys.path.insert(0, model_dir)

from yak_inference import YakClassifier

classifier = YakClassifier.from_pretrained(model_dir)
results = classifier.predict(
    "I need to transfer money to another account",
    candidate_labels=[
        "transfer money",
        "check account balance",
        "report a lost card",
        "cancel a transaction",
    ],
)
print(results)
```

The return value is a list of dictionaries with `label` and `score` keys, sorted
from highest to lowest score. Scores are normalized over the candidate labels
provided in that call; they are not calibrated probabilities across all possible
labels.

## Model details

- Architecture: Yak candidate-label classifier
- Trainable parameters: 4,634,113 (4.63 million)
- Safetensors contains 4,896,257 float32 values, including the fixed sinusoidal
  positional-encoding buffer (262,144 values). This non-trainable buffer accounts
  for the difference from the Hub's rounded 4.9M size display.
- Weight tensor type: float32 (F32)
- Encoder: 4 transformer encoder blocks, hidden size 128, 4 attention heads
- Vocabulary: 30,005-token Yak BPE vocabulary
- Maximum packed input length: 2,048 tokens
- Text inputs are truncated to 96 tokens; premise and hypothesis inputs are
  truncated to 64 and 32 tokens, respectively; candidate labels are truncated to
  12 tokens by default.
- Candidate-label order shuffling is averaged over four passes by default.

## Intended use and limitations

Use this model to score a caller-supplied set of candidate labels for short text
classification or sentence-pair inference. The returned scores depend on the
candidate labels supplied, their wording, and the model's training data. They
should not be treated as calibrated confidence or as the sole basis for
high-impact decisions. Evaluate the model on data representative of your intended
use before deployment.

The model package identifies Banking77, CLINC, and SNLI as its supported tasks.
No benchmark results are claimed in this card.

## Files

- `model.safetensors` and `config.json`: model parameters and configuration
- `bpe.json`: tokenizer vocabulary and merges required by the model
- `label_metadata.json`: task label metadata
- `yak_inference.py`: standalone loader and `.predict()` API
- `yak_manifest.json`: inference limits and package metadata
- `requirements.txt`: Python runtime dependencies

## License

No license is specified in this model card. Confirm that you have the rights to
redistribute the model weights, tokenizer, and associated data, then add the
appropriate license metadata before reuse or redistribution.
