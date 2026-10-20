from yak.tokeniser.bpe import (
    tokenise_string,
    learn_vocab,
    BASE_ENGLISH_VOCAB
)

def test_tokenise_empty_string():
    assert tokenise_string(["a"], "") == []

def test_tokenise_single_token():
    assert tokenise_string(["a"], "a") == ["a"]

def test_tokenise_multiple_same_tokens():
    assert tokenise_string(["a"], "aa") == ["a", "a"]

def test_tokenise_unknown_token():
    assert tokenise_string(["a"], "b") == ["</UNKNOWN>"]

def test_tokenise_unknown_followed_by_known_token():
    assert tokenise_string(["a"], "ba") == ["</UNKNOWN>", "a"]

def test_tokenise_multiple_same_tokens():
    assert tokenise_string(["a", "b"], "ab") == ["a", "b"]

def test_tokenise_biggest_first():
    assert tokenise_string(["a", "b", "ab"], "ab") == ["ab"]

def test_learn_vocab_single_string():
    assert learn_vocab(["a"], "aa", iterations=1) == ["a", "aa"]

def test_learn_vocab_multiple_iterations():
    assert learn_vocab(["a"], "aaa", iterations=2) == ["a", "aa", "aaa"]

def test_tokenise_common_pairs():
    vocab = learn_vocab(["s", "d", "i", "n", "g", " "], "sing ding", iterations=2)
    tokens = tokenise_string(vocab, "sing ding")
    assert tokens == ["s", "ing", " ", "d", "ing"]

def test_tokenise_handle_word_endings():
    vocab = learn_vocab(BASE_ENGLISH_VOCAB, "sing ding ling", iterations=3)
    tokens = tokenise_string(vocab, "sing ding ling")
    assert tokens == ["s", "ing ", "d", "ing ", "l", "ing"]

def test_tokenise_different_string_from_learn():
    vocab = learn_vocab(BASE_ENGLISH_VOCAB, "sing ding ling", iterations=3)
    tokens = tokenise_string(vocab, "thing")
    assert tokens == ["t", "h", "ing"]
