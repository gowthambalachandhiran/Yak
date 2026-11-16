import re
import json
from tqdm.auto import tqdm


class BPE:
    def __init__(self, file=None, vocab_size=30000):
        self.file = file
        self.vocab_size = int(vocab_size)
        self.vocab = set()
        self.word_freq = {}
        self.merges = []          # ordered list of merged pairs
        self.tok2id = {}
        self.id2tok = {}
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    # ---------- training ----------
    def get_word_freq(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        for word in tqdm(words, desc="Counting words", unit="word"):
            new_word = ' '.join(list(word)) + ' </w>'
            if new_word not in self.word_freq:
                self.word_freq[new_word] = 1
            else:
                self.word_freq[new_word] += 1

    def get_pair(self):
        pair_dict = {}
        for word, freq in self.word_freq.items():
            symbols = word.split()
            for i in range(1, len(symbols)):
                pair = (symbols[i - 1], symbols[i])
                if pair not in pair_dict:
                    pair_dict[pair] = freq
                else:
                    pair_dict[pair] += freq
        return pair_dict

    def get_best_pair(self, pair_dict):
        return max(pair_dict, key=lambda k: (pair_dict[k], k))

    def merge_rule(self, best_pair):
        new_word_freq = {}
        bigram = re.escape(' '.join(best_pair))
        pattern = r'(?<!\S)' + bigram + r'(?!\S)'
        for word, freq in self.word_freq.items():
            new_word = re.sub(pattern, ''.join(best_pair), word)
            if new_word not in new_word_freq:
                new_word_freq[new_word] = freq
            else:
                new_word_freq[new_word] += freq
        self.word_freq = new_word_freq

    def build_vocab_maps(self):
        self.tok2id = {}
        self.id2tok = {}
        all_tokens = self.special_tokens + sorted(self.vocab)
        for i, tok in enumerate(all_tokens):
            self.tok2id[tok] = i
            self.id2tok[i] = tok

    def train(self):
        self.get_word_freq()
        for word in self.word_freq:
            for symbol in word.split():
                self.vocab.add(symbol)

        total_merges = max(self.vocab_size - len(self.vocab), 0)
        print(f"Unique words: {len(self.word_freq)} | "
              f"base symbols: {len(self.vocab)} | merges to learn: {total_merges}")

        with tqdm(total=total_merges, desc="Training BPE", unit="merge") as pbar:
            while len(self.vocab) < self.vocab_size:
                pair_dict = self.get_pair()
                if not pair_dict:          # nothing left to merge
                    print("\nNo more pairs to merge, stopping early.")
                    break
                max_key = self.get_best_pair(pair_dict)
                self.merge_rule(max_key)
                self.merges.append(max_key)
                self.vocab.add(''.join(max_key))

                pbar.update(1)
                pbar.set_postfix(vocab=len(self.vocab), last=''.join(max_key))

        self.build_vocab_maps()

    # ---------- encoding ----------
    def encode_word(self, word):
        word = ' '.join(list(word)) + ' </w>'
        for pair in self.merges:
            bigram = re.escape(' '.join(pair))
            pattern = r'(?<!\S)' + bigram + r'(?!\S)'
            word = re.sub(pattern, ''.join(pair), word)
        return word.split()

    def encode(self, text, show_progress=False):
        # split out special tokens first so "[MASK]" stays ONE token
        special_pattern = '(' + '|'.join(re.escape(t) for t in self.special_tokens) + ')'
        parts = re.split(special_pattern, text)

        tokens = []
        for part in parts:
            if part in self.special_tokens:
                tokens.append(part)
            else:
                words = part.split()
                if show_progress:
                    words = tqdm(words, desc="Encoding", unit="word")
                for word in words:
                    tokens.extend(self.encode_word(word))
        return tokens

    def tokens_to_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.tok2id:
                ids.append(self.tok2id[tok])
            else:
                ids.append(self.tok2id['[UNK]'])
        return ids

    def encode_ids(self, text, add_special_tokens=False, show_progress=False):
        ids = self.tokens_to_ids(self.encode(text, show_progress=show_progress))
        if add_special_tokens:
            ids = [self.tok2id['[CLS]']] + ids + [self.tok2id['[SEP]']]
        return ids

    def encode_pair(self, text_a, text_b):
        # [CLS] A [SEP] B [SEP]  + segment ids (0 for A part, 1 for B part)
        cls_id = self.tok2id['[CLS]']
        sep_id = self.tok2id['[SEP]']
        ids_a = self.tokens_to_ids(self.encode(text_a))
        ids_b = self.tokens_to_ids(self.encode(text_b))

        ids = [cls_id] + ids_a + [sep_id] + ids_b + [sep_id]
        segment_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)
        return ids, segment_ids

    # ---------- decoding ----------
    def decode_ids(self, ids, skip_special_tokens=False):
        text = ''
        for i in ids:
            tok = self.id2tok[i]
            if tok in self.special_tokens:
                if not skip_special_tokens:
                    text += ' ' + tok + ' '
            else:
                text += tok
        text = text.replace('</w>', ' ')
        return ' '.join(text.split())     # collapse extra spaces

    # ---------- save / load ----------
    def save(self, path):
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'merges': [list(p) for p in self.merges],
            'tok2id': self.tok2id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved tokenizer to {path} ({len(self.tok2id)} tokens, {len(self.merges)} merges)")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.merges = [tuple(p) for p in data['merges']]
        self.tok2id = data['tok2id']
        self.id2tok = {}
        for tok, i in self.tok2id.items():
            self.id2tok[i] = tok
        self.vocab = set(self.tok2id) - set(self.special_tokens)