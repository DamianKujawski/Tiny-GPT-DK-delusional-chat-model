"""Train a small LLM on instruction-response JSONL data."""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from encoder import BPEEncoder
from model import DKmodel, DKmodelConfig


def load_examples(path, encoder, block_size):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"Invalid JSON on line {line_number}: {error}"
                )

            instruction = record.get("instruction")
            response = record.get("response")

            if not isinstance(instruction, str) or not isinstance(response, str):
                continue
            if not instruction.strip() or not response.strip():
                continue

            # Keep the prompt format consistent with sample.py
            prompt = f"User: {instruction.strip()}\nAssistant:"
            prefix_ids = encoder.encode(prompt)
            answer_ids = encoder.encode(
                f" {response.strip()}\n<|endoftext|>"
            )

            if not answer_ids:
                continue

            # Keep the answer and as much of the prompt as fits
            max_length = block_size + 1
            answer_ids = answer_ids[:max_length - 1]
            prefix_budget = max_length - len(answer_ids)
            prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget > 0 else []

            ids = prefix_ids + answer_ids
            if len(ids) < 2:
                continue

            # Inputs and next-token targets
            x = ids[:-1]
            y = ids[1:]
            prefix_length = len(prefix_ids)

            # Ignore targets that belong to the user prompt
            y[:max(0, prefix_length - 1)] = [-100] * max(
                0, prefix_length - 1
            )

            # Pad to a fixed length and ignore padded target positions
            padding = block_size - len(x)
            if padding > 0:
                x += [0] * padding
                y += [-100] * padding

            records.append((
                torch.tensor(x[:block_size], dtype=torch.long),
                torch.tensor(y[:block_size], dtype=torch.long),
            ))

    if len(records) < 2:
        raise SystemExit(
            "At least 2 valid records with 'instruction' and 'response' "
            "fields are required in the JSONL file."
        )

    return records


def get_batch(data, batch_size, device):
    indices = torch.randint(len(data), (batch_size,)).tolist()
    x = torch.stack([data[i][0] for i in indices]).to(device)
    y = torch.stack([data[i][1] for i in indices]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, device, steps=10):
    model.eval()
    losses = []

    for _ in range(steps):
        x, y = get_batch(data, batch_size, device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="data.txt")
    parser.add_argument("--out", default="checkpoint_v4.pt")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.isfile(args.text):
        raise SystemExit(f"File not found: {args.text}")

    encoder = BPEEncoder()
    examples = load_examples(args.text, encoder, args.block_size)
    random.shuffle(examples)

    split = max(1, int(0.9 * len(examples)))
    split = min(split, len(examples) - 1)
    train_data = examples[:split]
    val_data = examples[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = DKmodelConfig(
        vocab_size=encoder.vocab_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.width,
        dropout=args.dropout,
    )
    model = DKmodel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Valid examples: {len(examples):,}")
    print(f"Training examples: {len(train_data):,}")
    print(f"Validation examples: {len(val_data):,}")
    print(f"Vocabulary: {encoder.vocab_size:,}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training from random initialization.")

    start = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        xb, yb = get_batch(train_data, args.batch_size, device)
        logits, _ = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_loss = estimate_loss(
                model, train_data, args.batch_size, device
            )
            val_loss = estimate_loss(
                model, val_data, args.batch_size, device
            )
            print(
                f"step {step:>6}/{args.steps} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )

    payload = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "training_args": vars(args),
        "encoder": "bytelevel-bpe-v1",
    }
    torch.save(payload, args.out)

    print(f"Saved checkpoint: {args.out}")
    print(f"Elapsed: {time.time() - start:.1f}s")
    print(
        "For a test, run: "
        'python sample.py --prompt "User: Why can camels survive '
        'for long without water?\\nAssistant:"'
    )


if __name__ == "__main__":
    main()
