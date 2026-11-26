import argparse
import torch

from encoder import BPEEncoder
from model import DKmodel, DKmodelConfig


def load_model(path, device):
    # Checkpoints are assumed to be files created by this project's train.py
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = DKmodelConfig(**checkpoint["config"])
    model = DKmodel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--prompt", default="Morning! ")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.checkpoint, device)
    encoder = BPEEncoder()
    eos_token_id = encoder.tokenizer.token_to_id("<|endoftext|>")

    def answer(prompt):
        # Convert escaped newline markers into actual newlines
        prompt = prompt.replace("`n", "\n").replace("\\n", "\n")

        # Match the prompt format used during training
        if not prompt.lstrip().startswith("User:"):
            prompt = f"User: {prompt.strip()}\nAssistant:"
        elif "\nAssistant:" not in prompt:
            prompt = prompt.rstrip() + "\nAssistant:"

        ids = encoder.encode(prompt)
        if not ids:
            ids = encoder.encode("User: \nAssistant:")

        context = torch.tensor([ids], dtype=torch.long, device=device)
        result = model.generate(
            context,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_token_id,
        )[0].tolist()

        # Remove the end-of-text token from the displayed continuation
        generated = result[len(ids):]
        if eos_token_id is not None and eos_token_id in generated:
            generated = generated[:generated.index(eos_token_id)]

        print(encoder.decode(generated), flush=True)

    if args.interactive:
        while True:
            try:
                prompt = input("\nTy> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            answer(prompt)
    else:
        answer(args.prompt)


if __name__ == "__main__":
    main()
