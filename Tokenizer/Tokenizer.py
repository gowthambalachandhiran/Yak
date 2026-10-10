import re
import json


class BPE:
    def __init__(self, file=None, vocab_size=30000):
        self.file = file
        self.vocab_size = vocab_size
        self.vocab = set()
        self.word_freq = {}
        self.merges = []          # ordered list of merged pairs
        self.tok2id = {}
        self.id2tok = {}
        self.special_tokens = ['<pad>', '<unk>']

    # ---------- training ----------
    def get_word_freq(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        for word in words:
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

        while len(self.vocab) < self.vocab_size:
            pair_dict = self.get_pair()
            if not pair_dict:
                break
            max_key = self.get_best_pair(pair_dict)
            self.merge_rule(max_key)
            self.merges.append(max_key)
            self.vocab.add(''.join(max_key))

        self.build_vocab_maps()

    # ---------- encoding / decoding ----------
    def encode_word(self, word):
        word = ' '.join(list(word)) + ' </w>'
        for pair in self.merges:
            bigram = re.escape(' '.join(pair))
            pattern = r'(?<!\S)' + bigram + r'(?!\S)'
            word = re.sub(pattern, ''.join(pair), word)
        return word.split()

    def encode(self, text):
        tokens = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
        return tokens

    def tokens_to_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.tok2id:
                ids.append(self.tok2id[tok])
            else:
                ids.append(self.tok2id['<unk>'])
        return ids

    def encode_ids(self, text):
        return self.tokens_to_ids(self.encode(text))

    def decode_ids(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.id2tok[i])
        text = ''.join(tokens)
        return text.replace('</w>', ' ').strip()

    # ---------- save / load ----------
    def save(self, path):
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'merges': [list(p) for p in self.merges],   # tuples -> lists for json
            'tok2id': self.tok2id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.merges = [tuple(p) for p in data['merges']]   # lists -> tuples
        self.tok2id = data['tok2id']
        self.id2tok = {}
        for tok, i in self.tok2id.items():
            self.id2tok[i] = tok                            # rebuild, don't store
        self.vocab = set(self.tok2id) - set(self.special_tokens)