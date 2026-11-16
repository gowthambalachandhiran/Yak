#FastBpe.py
"""Fast, cached encoder on top of the trained BPE in Tokenizer/Tokenizer.py, plus text normalisation.

BPE.encode_word() replays all ~25,000 merges with a regex for every word, which is far too slow for
fine-tuning. FastBPE gives identical tokens by always applying the lowest-ranked merge that is still
allowed (rank above the last one applied) and caches every word it has seen.
"""
import re

_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

_APOS = re.compile(r"(?<=\w)'(?=\w)")                 # don't -> don 't  (WikiText style)
_PUNCT = re.compile(r"([.,!?;:()\"\[\]{}$%&*+/<=>@^_`|~\\-])")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Space out punctuation the way the WikiText pre-training corpus was written.

    Case is kept (WikiText keeps case). "card?" becomes "card ?" so the tokenizer sees the same
    pieces it saw during pre-training instead of an unseen "card?" word.
    """
    text = _APOS.sub(" '", text)
    text = _PUNCT.sub(r" \1 ", text)
    return _SPACES.sub(" ", text).strip()


def normalize_label(name: str) -> str:
    """'card_arrival' -> 'card arrival', 'Refund_not_showing_up' -> 'Refund not showing up'."""
    return _SPACES.sub(" ", name.replace("_", " ").replace("-", " ")).strip()


class FastBPE:
    def __init__(self, bpe):
        """bpe: a loaded Tokenizer.Tokenizer.BPE (call bpe.load(path) first)."""
        self.bpe = bpe
        self.tok2id = bpe.tok2id
        self.rank = {tuple(p): i for i, p in enumerate(bpe.merges)}
        self.unk_id = self.tok2id["[UNK]"]
        self.pad_id = self.tok2id["[PAD]"]
        self.cls_id = self.tok2id["[CLS]"]
        self.sep_id = self.tok2id["[SEP]"]
        self.mask_id = self.tok2id["[MASK]"]
        self.vocab_size = len(self.tok2id)
        self._word_cache = {}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in _SPECIALS) + ")")

    # ------------------------------------------------------------------ words
    def encode_word(self, word: str):
        hit = self._word_cache.get(word)
        if hit is not None:
            return hit
        symbols = list(word) + ["</w>"]
        last = -1
        while len(symbols) > 1:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                r = self.rank.get((symbols[i], symbols[i + 1]))
                if r is not None and r > last and (best_rank is None or r < best_rank):
                    best, best_rank = (symbols[i], symbols[i + 1]), r
            if best is None:
                break
            merged, out, i = best[0] + best[1], [], 0
            while i < len(symbols):                      # left to right, non-overlapping (like the regex)
                if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols, last = out, best_rank
        self._word_cache[word] = symbols
        return symbols

    # ------------------------------------------------------------------ text
    def encode_tokens(self, text: str):
        tokens = []
        for part in self._special_re.split(text):
            if part in _SPECIALS:
                tokens.append(part)
            else:
                for word in part.split():
                    tokens.extend(self.encode_word(word))
        return tokens

    def encode_ids(self, text: str, normalize: bool = True):
        """Text -> token ids, no [CLS]/[SEP]. Unknown pieces become [UNK]."""
        if normalize:                                   # keep "[SEP]" etc. intact, normalise the rest
            text = " ".join(p if p in _SPECIALS else normalize_text(p) for p in self._special_re.split(text))
        return [self.tok2id.get(t, self.unk_id) for t in self.encode_tokens(text)]

    def unk_rate(self, texts, normalize: bool = True):
        total = unk = 0
        for t in texts:
            ids = self.encode_ids(t, normalize)
            total += len(ids)
            unk += sum(1 for i in ids if i == self.unk_id)
        return unk / max(total, 1), total