import json
import os
import re
from transformers import PreTrainedTokenizer

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
APOS = re.compile(r"(?<=\w)'(?=\w)")
PUNCT = re.compile(r"([.,!?;:()\"\[\]{}$%&*+/<=>@^_`|~\\-])")
SPACES = re.compile(r"\s+")

def normalize_text(text):
    text = APOS.sub(" '", text)
    return SPACES.sub(" ", PUNCT.sub(r" \1 ", text)).strip()

def normalize_label(text):
    return SPACES.sub(" ", text.replace("_", " ").replace("-", " ")).strip()


class YakTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "bpe.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, **kwargs):
        with open(vocab_file, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_file = vocab_file
        self.merges = [tuple(x) for x in data["merges"]]
        self.ranks = {x: i for i, x in enumerate(self.merges)}
        self.encoder = {k: int(v) for k, v in data["tok2id"].items()}
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.cache = {}
        self.specials = data.get("special_tokens", SPECIALS)
        self.special_re = re.compile("(" + "|".join(re.escape(x) for x in self.specials) + ")")
        super().__init__(vocab_file=vocab_file, pad_token="[PAD]", unk_token="[UNK]",
                         cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", **kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder)

    def _word(self, word):
        if word in self.cache:
            return self.cache[word]
        symbols, last = list(word) + ["</w>"], -1
        while len(symbols) > 1:
            best, rank = None, None
            for i in range(len(symbols)-1):
                r = self.ranks.get((symbols[i], symbols[i+1]))
                if r is not None and r > last and (rank is None or r < rank):
                    best, rank = (symbols[i], symbols[i+1]), r
            if best is None:
                break
            merged, out, i = best[0] + best[1], [], 0
            while i < len(symbols):
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == best:
                    out.append(merged); i += 2
                else:
                    out.append(symbols[i]); i += 1
            symbols, last = out, rank
        self.cache[word] = symbols
        return symbols

    def _tokenize(self, text, **kwargs):
        out = []
        for part in self.special_re.split(text):
            if part in self.specials:
                out.append(part)
            else:
                for word in normalize_text(part).split():
                    out.extend(self._word(word))
        return out

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.decoder.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join("".join(tokens).replace("</w>", " ").split())

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        name = (filename_prefix + "-" if filename_prefix else "") + "bpe.json"
        dest = os.path.join(save_directory, name)
        if os.path.abspath(dest) != os.path.abspath(self.vocab_file):
            with open(self.vocab_file, encoding="utf-8") as src, open(dest, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        return (dest,)

    def pack_candidates(self, text, candidate_labels, max_text_tokens=96, max_label_tokens=12):
        text_ids = self.convert_tokens_to_ids(self._tokenize(normalize_text(text)))[:max_text_tokens]
        ids, positions = [self.cls_token_id] + text_ids + [self.sep_token_id], []
        for label in candidate_labels:
            label = normalize_label(label)
            label_ids = self.convert_tokens_to_ids(self._tokenize(label))[:max_label_tokens]
            if not label_ids:
                label_ids = [self.unk_token_id]
            positions.append(len(ids))
            ids.extend(label_ids + [self.sep_token_id])
        return {"input_ids": ids, "label_positions": positions}

    def save_pretrained(self, save_directory, **kwargs):
        return super().save_pretrained(save_directory, **kwargs)
