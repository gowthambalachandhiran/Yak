import argparse
import json
from pathlib import Path
import torch
from huggingface.configuration_yak import YakConfig
from huggingface.modeling_yak import YakForLabelScoring
from huggingface.tokenization_yak import YakTokenizer


def read_checkpoint(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("Expected a PyTorch checkpoint dictionary")
    state = obj
    for key in ("model", "model_state", "model_state_dict", "state_dict"):
        if isinstance(obj.get(key), dict):
            state = obj[key]
            break
    clean = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            for prefix in ("module.", "_orig_mod."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            clean[key] = value
    return obj, clean


def main():
    parser = argparse.ArgumentParser(description="Export Yak classifier to Hugging Face")
    parser.add_argument("--checkpoint", default=None, help="Exact .pt path; omit to auto-discover")
    parser.add_argument("--checkpoint_dir", default="runs")
    parser.add_argument("--output_dir", default="release/yak-classifier")
    parser.add_argument("--tokenizer_file", default="Tokenizer/TokenFile/bpe.json")
    parser.add_argument("--label_metadata", default="Data/processed/label_metadata.json")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        candidates = sorted(Path(args.checkpoint_dir).rglob("*.pt"))
        preferred = [x for x in candidates if x.name.lower() in ("best.pt", "best_model.pt")]
        if len(preferred) == 1:
            checkpoint = preferred[0]
        elif len(candidates) == 1:
            checkpoint = candidates[0]
        else:
            listing = "\n".join(str(x) for x in candidates) or "(no .pt files found)"
            raise RuntimeError("Could not select one checkpoint. Found:\n" + listing +
                               "\nPass --checkpoint with the exact file path.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    with open(args.tokenizer_file, encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    with open(args.label_metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    payload, state = read_checkpoint(checkpoint)

    embedding = state.get("embed.tok_emb.weight")
    if embedding is None:
        raise KeyError("Missing embed.tok_emb.weight: this is not a YakClassifier checkpoint")
    vocab_size, width = embedding.shape
    if vocab_size != len(tokenizer_data["tok2id"]) or width != args.d_model:
        raise ValueError(f"Checkpoint embedding shape {(vocab_size, width)} does not match "
                         f"tokenizer/model {(len(tokenizer_data['tok2id']), args.d_model)}")
    block_ids = {int(k.split(".")[1]) for k in state
                 if k.startswith("blocks.") and k.split(".")[1].isdigit()}
    if len(block_ids) != args.num_layers:
        raise ValueError(f"Checkpoint has {len(block_ids)} blocks; expected {args.num_layers}")
    if "scorer.weight" not in state or "scorer.bias" not in state:
        raise KeyError("Classifier scorer weights are missing; this may be an MLM checkpoint")

    specials = tokenizer_data.get("special_tokens", ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    token_ids = tokenizer_data["tok2id"]
    config = YakConfig(
        vocab_size=vocab_size, d_model=width, nhead=args.nhead, num_layers=args.num_layers,
        max_seq_len=args.max_seq_len, dropout=args.dropout,
        pad_token_id=token_ids[specials[0]], cls_token_id=token_ids[specials[2]],
        sep_token_id=token_ids[specials[3]],
        label_vocabularies=metadata.get("label_vocabularies", {}),
        held_out=payload.get("held_out", {}),
        auto_map={
            "AutoConfig": "configuration_yak.YakConfig",
            "AutoModel": "modeling_yak.YakForLabelScoring",
            "AutoModelForSequenceClassification": "modeling_yak.YakForLabelScoring",
            "AutoTokenizer": "tokenization_yak.YakTokenizer",
        },
    )
    model = YakForLabelScoring(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch. Missing={missing[:8]}, unexpected={unexpected[:8]}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = YakTokenizer(args.tokenizer_file)
    tokenizer.save_pretrained(output)

    # Required by Transformers' remote-code AutoTokenizer loader.
    tokenizer_config_path = output / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["auto_map"] = {"AutoTokenizer": ["tokenization_yak.YakTokenizer", None]}
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")

    card = (
        "---\nlibrary_name: transformers\ntags:\n- yak\n- text-classification\n"
        "- candidate-label-scoring\n---\n\n# Yak candidate-label classifier\n\n"
        "Yak scores text-described candidate labels. For each candidate, it applies one shared "
        "linear scorer to the hidden state at that label's first token. Logits follow candidate order.\n\n"
        f"Hidden size: {width}; attention heads: {args.nhead}; encoder blocks: {args.num_layers}; "
        f"maximum sequence length: {args.max_seq_len}.\n\n"
        "Use YakTokenizer.pack_candidates(text, candidate_labels), then pass input_ids, "
        "label_positions and label_mask to YakForLabelScoring. The source .pt checkpoint is not included.\n"
    )
    (output / "README.md").write_text(card, encoding="utf-8")
    print(f"Exported {checkpoint} to {output.resolve()}")


if __name__ == "__main__":
    main()
