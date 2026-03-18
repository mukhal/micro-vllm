"""
Minimal tokenizer: load tiktoken-style encoding from disk for inference.
"""
import os
import pickle
from functools import lru_cache


def load_tokenizer(tokenizer_dir: str):
    """Load tokenizer from a directory containing tokenizer.pkl (nanochat-style)."""
    path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(path, "rb") as f:
        enc = pickle.load(f)
    return Tokenizer(enc)


class Tokenizer:
    """Thin wrapper over tiktoken Encoding for encode/decode and special tokens."""

    def __init__(self, enc):
        self.enc = enc
        try:
            self._bos_token_id = enc.encode_single_token("<|bos|>")
        except Exception:
            try:
                self._bos_token_id = enc.encode_single_token("<|endoftext|>")
            except Exception:
                raise ValueError("Tokenizer has no <|bos|> or <|endoftext|> token")

    def get_vocab_size(self):
        return self.enc.n_vocab

    @lru_cache(maxsize=32)
    def encode_special(self, text: str):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self._bos_token_id

    def encode(self, text: str, prepend=None, append=None):
        if isinstance(text, str):
            ids = list(self.enc.encode_ordinary(text))
            if prepend is not None:
                pid = prepend if isinstance(prepend, int) else self.encode_special(prepend)
                ids.insert(0, pid)
            if append is not None:
                aid = append if isinstance(append, int) else self.encode_special(append)
                ids.append(aid)
            return ids
        raise ValueError(f"Expected str, got {type(text)}")

    def decode(self, ids):
        return self.enc.decode(ids)
