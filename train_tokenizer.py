
import os
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["data.txt"],
    vocab_size=2000,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
)

os.makedirs("tokenizer", exist_ok=True)

tokenizer.save_model("tokenizer")

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print("Tokenizer files saved successfully.")
