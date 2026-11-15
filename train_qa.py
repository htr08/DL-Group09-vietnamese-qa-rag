
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class QADataset(Dataset):
    def __init__(self, path):
        self.features = []

        with Path(path).open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    feature = json.loads(line)
                    self.features.append(feature)

        if not self.features:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]

        return {
            "input_ids": feature["input_ids"],
            "start_positions": feature["start_positions"],
            "end_positions": feature["end_positions"],
        }


def collate_fn(batch, pad_token_id):
    max_length = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    start_positions = []
    end_positions = []

    for item in batch:
        ids = item["input_ids"]
        padding_length = max_length - len(ids)

        input_ids.append(ids + [pad_token_id] * padding_length)
        attention_mask.append([1] * len(ids) + [0] * padding_length)
        start_positions.append(item["start_positions"])
        end_positions.append(item["end_positions"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "start_positions": torch.tensor(start_positions, dtype=torch.long),
        "end_positions": torch.tensor(end_positions, dtype=torch.long),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            batch_size = batch["input_ids"].size(0)
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    train_dataset = QADataset(args.train_file)
    dev_dataset = QADataset(args.dev_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer.pad_token_id
        ),
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer.pad_token_id
        ),
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / max(1, warmup_steps)
            if step < warmup_steps
            else max(
                0.0,
                (total_steps - step) / max(1, total_steps - warmup_steps)
            )
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dev_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_train_loss = 0.0
        total_train_samples = 0

        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            batch_size = batch["input_ids"].size(0)

            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            scaler.scale(
                loss / args.gradient_accumulation_steps
            ).backward()

            should_update = (
                (step + 1) % args.gradient_accumulation_steps == 0
                or step + 1 == len(train_loader)
            )

            if should_update:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"| Step {step + 1}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f}"
                )

        train_loss = total_train_loss / total_train_samples
        dev_loss = evaluate(model, dev_loader, device)

        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Dev loss: {dev_loss:.4f}")
        print(f"Learning rate: {current_lr:.8f}")

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "learning_rate": current_lr,
        }

        history.append(epoch_result)

        with (output_dir / "history.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history, file, ensure_ascii=False, indent=2)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

            best_dir = output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            print(f"Saved best model: {best_dir}")

    print(f"\nBest Dev Loss: {best_dev_loss:.4f}")
    print(f"Training completed. Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="vinai/phobert-base",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/qa_features/train.jsonl",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="data/processed/qa_features/dev.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/phobert_qa",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
