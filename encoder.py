from tokenizers import ByteLevelBPETokenizer


class BPEEncoder:
    def __init__(
        self,
        vocab_path: str = "tokenizer/vocabulary.json",
        merges_path: str = "tokenizer/tokens.txt",
    ):
        self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        self.tokenizer.add_special_tokens(["<|endoftext|>"])
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens) -> str:
        # Handle PyTorch tensors or NumPy arrays
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        # Unwrap batch dimension if a 2D list is passed (e.g., [[1, 2, 3]])
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]

        return self.tokenizer.decode([int(t) for t in tokens])


# Lazy-load a shared instance to avoid reading vocab files on every call
_default_encoder = None


def _get_encoder() -> BPEEncoder:
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = BPEEncoder()
    return _default_encoder


def encode(text: str) -> list[int]:
    return _get_encoder().encode(text)


def decode(tokens) -> str:
    return _get_encoder().decode(tokens)
