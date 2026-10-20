from collections import Counter
from string import ascii_letters, punctuation, digits, whitespace

BASE_ENGLISH_VOCAB = list(ascii_letters + punctuation + digits + whitespace)

def tokenise_string(vocab, string):
    if string == "":
        return []
    for token in sorted(vocab, key=lambda k: len(k), reverse=True):
        if string.startswith(token):
            return [token] + tokenise_string(vocab, string[len(token):])
    return ["</UNKNOWN>"] + tokenise_string(vocab, string[1:])

def learn_vocab(initial_vocab, dataset, iterations=32000):
    if iterations == 0:
        return initial_vocab
    tokens = tokenise_string(initial_vocab, dataset)
    if len(tokens) == 1:
        return initial_vocab
    token_pairs = zip(tokens, tokens[1:])
    counts = Counter(token_pairs)
    merge_tokens, _ = counts.most_common(1)[0]
    new_token = "".join(merge_tokens)
    return learn_vocab(initial_vocab + [new_token], dataset, iterations-1)
