# Yak Hugging Face export

This directory contains the Hugging Face integration for the fine-tuned Yak candidate-label classifier.

## 1. Install dependencies

From the repository root:

    pip install -r requirements-huggingface.txt

## 2. Convert your local checkpoint

Your local checkpoint directory is runs. The converter searches that directory recursively and selects a unique .pt file. If several checkpoints exist, it prefers a unique best.pt or best_model.pt. Otherwise, pass the exact file.

    python -m huggingface.convert_checkpoint --output_dir release/yak-classifier

Or specify the checkpoint explicitly:

    python -m huggingface.convert_checkpoint --checkpoint "runs/mixture/best.pt" --output_dir "release/yak-classifier"

The converter does not upload or commit your .pt file. It creates a standard export directory with model weights, config, tokenizer files and a model card.

The defaults follow the fine-tuning notebook: hidden size 128, 4 attention heads, 4 encoder blocks, maximum sequence length 2048, and dropout 0.1. If your training run used different values, override --d_model, --nhead, --num_layers, --max_seq_len and --dropout.

## 3. Run inference

    import torch
    from huggingface.modeling_yak import YakForLabelScoring
    from huggingface.tokenization_yak import YakTokenizer

    path = "release/yak-classifier"
    tokenizer = YakTokenizer(path + "/bpe.json")
    model = YakForLabelScoring.from_pretrained(path)
    model.eval()

    candidates = ["refund_not_showing_up", "card_arrival", "cash_withdrawal_charge"]
    packed = tokenizer.pack_candidates("My card has not arrived yet", candidates)

    input_ids = torch.tensor([packed["input_ids"]])
    label_positions = torch.tensor([packed["label_positions"]])
    label_mask = torch.ones_like(label_positions, dtype=torch.bool)

    with torch.no_grad():
        logits = model(input_ids=input_ids,
                       label_positions=label_positions,
                       label_mask=label_mask).logits
        probabilities = logits.softmax(dim=-1)[0]

    print(sorted(zip(candidates, probabilities.tolist()), key=lambda x: -x[1]))

The returned logits follow the candidate-label order supplied to pack_candidates. Yak is a candidate-label scoring model, not a conventional fixed-class softmax head.

## 4. Publish to Hugging Face Hub

After reviewing the generated model card and repository license:

    huggingface-cli login
    cd release/yak-classifier
    huggingface-cli upload YOUR_USERNAME/YOUR_MODEL_NAME .

Keep the original training checkpoint private/local unless you intentionally want to publish it. Upload the converted export directory instead.
