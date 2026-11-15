# Yak Hugging Face export

Run the converter from the Yak repository root:

    pip install -r requirements-huggingface.txt
    python -m huggingface.convert_checkpoint --checkpoint "runs/mixture/best.pt" --output_dir "release/yak-classifier"

Use the actual .pt filename under your runs directory if it differs. The checkpoint is read locally and is not committed or uploaded by the converter.

The default architecture matches the fine-tuning notebook: d_model 128, 4 attention heads, 4 encoder blocks, max sequence length 2048, dropout 0.1. Override the command-line options if your run used different settings.

The export directory contains model weights, config, tokenizer files and a model card. For inference, use YakTokenizer.pack_candidates(text, candidate_labels), create a batch with input_ids, label_positions and a boolean label_mask, then read model(...).logits. Softmax over the candidate dimension gives candidate probabilities.
