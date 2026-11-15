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
    p = argparse.ArgumentParser(description="Export Yak classifier to Hugging Face format")
    p.add_argument("--checkpoint", default=None, help="Exact checkpoint path; omit to auto-discover .pt files under --checkpoint_dir")\n    p.add_argument("--checkpoint_dir", default="runs")
    p.add_argument("--output_dir", default="release/yak-classifier")
    p.add_argument("--tokenizer_file", default="Tokenizer/TokenFile/bpe.json")
    p.add_argument("--label_metadata", default="Data/processed/label_metadata.json")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with open(args.tokenizer_file, encoding="utf-8") as f:
        tok = json.load(f)
    with open(args.label_metadata, encoding="utf-8") as f:
        labels = json.load(f)
    payload, state = read_checkpoint(ckpt_path)
    embedding = state.get("embed.tok_emb.weight")
    if embedding is None:
        raise KeyError("Missing embed.tok_emb.weight; this does not look like a Yak classifier checkpoint")
    vocab_size, width = embedding.shape
    if vocab_size != len(tok["tok2id"]) or width != args.d_model:
        raise ValueError(f"Checkpoint embedding shape is {(vocab_size, width)}; tokenizer/model expect "
                         f"{(len(tok['tok2id']), args.d_model)}")
    blocks = {int(k.split(".")[1]) for k in state if k.startswith("blocks.") and k.split(".")[1].isdigit()}
    if len(blocks) != args.num_layers:
        raise ValueError(f"Checkpoint has {len(blocks)} blocks; pass the matching --num_layers")
    if "scorer.weight" not in state or "scorer.bias" not in state:
        raise KeyError("Classifier scorer weights are missing; selected checkpoint may be an MLM checkpoint")

    specials = tok.get("special_tokens", ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    ids = tok["tok2id"]
    config = YakConfig(
        vocab_size=vocab_size, d_model=width, nhead=args.nhead, num_layers=args.num_layers,
        max_seq_len=args.max_seq_len, dropout=args.dropout,
        pad_token_id=ids[specials[0]], cls_token_id=ids[specials[2]], sep_token_id=ids[specials[3]],
        label_vocabularies=labels.get("label_vocabularies", {}),
        held_out=payload.get("held_out", {}),
        auto_map={"AutoConfig": "configuration_yak.YakConfig",
                  "AutoModel": "modeling_yak.YakForLabelScoring",
                  "AutoModelForSequenceClassification": "modeling_yak.YakForLabelScoring",
                  "AutoTokenizer": "tokenization_yak.YakTokenizer"})
    model = YakForLabelScoring(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch. Missing={missing[:8]}, unexpected={unexpected[:8]}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    YakTokenizer(args.tokenizer_file).save_pretrained(out)
    card = ("---\nlibrary_name: transformers\ntags:\n- yak\n- text-classification\n"
            "- candidate-label-scoring\n---\n\n# Yak candidate-label classifier\n\n"
            "Yak scores text-described candidate labels. For each candidate, the model reads the packed "
            "input and applies one shared linear scorer to the hidden state at that label's first token. "
            "The logits follow the candidate order supplied at inference time.\n\n"
            f"Hidden size: {width}; attention heads: {args.nhead}; encoder blocks: {args.num_layers}; "
            f"maximum sequence length: {args.max_seq_len}.\n\n"
            "Use YakTokenizer.pack_candidates(text, candidate_labels), then pass input_ids, "
            "label_positions and label_mask to YakForLabelScoring. The original .pt checkpoint is not included.\n")
    (out / "README.md").write_text(card, encoding="utf-8")
    print(f"Export complete: {out.resolve()}")


if __name__ == "__main__":
    main()
