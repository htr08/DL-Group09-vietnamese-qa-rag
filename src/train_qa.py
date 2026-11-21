
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

try:
    from .tracking import init_run
except ImportError:
    from tracking import init_run
from src.metrics import exact_match_score, f1_score


class QADataset(Dataset):
    def __init__(self, path):
        self.features = []

        with Path(path).open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.features.append(json.loads(line))

        if not self.features:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class ExtractiveQAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(
            getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        )
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = self.qa_outputs(
            self.dropout(outputs.last_hidden_state)
        )

        return logits.split(1, dim=-1)[0].squeeze(-1), \
            logits.split(1, dim=-1)[1].squeeze(-1)


def collate_fn(batch, pad_token_id):
    max_length = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    start_positions = []
    end_positions = []
    ids = []
    impossible = []

    for item in batch:
        ids_list = item["input_ids"]
        padding_length = max_length - len(ids_list)

        input_ids.append(
            ids_list + [pad_token_id] * padding_length
        )
        attention_mask.append(
            [1] * len(ids_list) + [0] * padding_length
        )
        start_positions.append(item["start_positions"])
        end_positions.append(item["end_positions"])
        ids.append(item["id"])
        impossible.append(item["is_impossible"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "start_positions": torch.tensor(start_positions, dtype=torch.long),
        "end_positions": torch.tensor(end_positions, dtype=torch.long),
        "ids": ids,
        "is_impossible": impossible,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(start_logits, end_logits, start_positions, end_positions):
    start_loss = nn.functional.cross_entropy(
        start_logits,
        start_positions,
    )
    end_loss = nn.functional.cross_entropy(
        end_logits,
        end_positions,
    )

    return (start_loss + end_loss) / 2


def decode_feature(
    input_ids,
    attention_mask,
    start_logits,
    end_logits,
    tokenizer,
    max_answer_length,
):
    valid_length = int(attention_mask.sum().item())

    input_ids = input_ids[:valid_length].tolist()
    start_logits = start_logits.float()
    end_logits = end_logits.float()

    context_start = None

    for i in range(len(input_ids) - 1):
        if input_ids[i] == 2 and input_ids[i + 1] == 2:
            context_start = i + 2
            break

    if context_start is None:
        raise ValueError("Cannot locate context separator")

    context_end = valid_length - 1

    null_score = (
        start_logits[0].item() + end_logits[0].item()
    )

    best_score = float("-inf")
    best_start = 0
    best_end = 0

    for start in range(context_start, context_end):
        max_end = min(
            context_end,
            start + max_answer_length,
        )

        for end in range(start, max_end):
            score = (
                start_logits[start].item()
                + end_logits[end].item()
            )

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    if best_score == float("-inf"):
        return "", null_score, best_score

    answer = tokenizer.decode(
        input_ids[best_start:best_end + 1],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return answer, null_score, best_score


def evaluate(
    model,
    loader,
    device,
    tokenizer,
    gold_data,
    max_answer_length,
    null_threshold,
    use_amp,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    predictions = defaultdict(
        lambda: {
            "best_answer": "",
            "best_score": float("-inf"),
            "null_score": float("inf"),
        }
    )

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                start_logits, end_logits = model(
                    input_ids,
                    attention_mask,
                )

                loss = compute_loss(
                    start_logits,
                    end_logits,
                    start_positions,
                    end_positions,
                )

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for i, question_id in enumerate(batch["ids"]):
                answer, null_score, answer_score = decode_feature(
                    input_ids[i],
                    attention_mask[i],
                    start_logits[i],
                    end_logits[i],
                    tokenizer,
                    max_answer_length,
                )

                item = predictions[question_id]

                item["null_score"] = min(
                    item["null_score"],
                    null_score,
                )

                if answer_score > item["best_score"]:
                    item["best_score"] = answer_score
                    item["best_answer"] = answer

    scores = {
        "all": {"em": [], "f1": []},
        "answerable": {"em": [], "f1": []},
        "impossible": {"em": [], "f1": []},
    }

    for row in gold_data:
        question_id = row["id"]
        item = predictions.get(question_id)

        if item is None:
            raise ValueError(
                f"Missing prediction for question {question_id}"
            )

        score_diff = item["null_score"] - item["best_score"]

        prediction = (
            ""
            if score_diff > null_threshold
            else item["best_answer"]
        )

        if row["is_impossible"]:
            references = [""]
            category = "impossible"
        else:
            references = row["answers"]["text"]
            category = "answerable"

        em = float(exact_match_score(prediction, references))
        f1 = f1_score(prediction, references)

        scores["all"]["em"].append(em)
        scores["all"]["f1"].append(f1)
        scores[category]["em"].append(em)
        scores[category]["f1"].append(f1)

    result = {
        "dev_loss": total_loss / total_samples,
    }

    for category, values in scores.items():
        for metric, scores_list in values.items():
            result[f"{category}_{metric}"] = (
                float(np.mean(scores_list))
                if scores_list
                else 0.0
            )

    return result


def load_gold_data(path):
    rows = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Empty gold dataset: {path}")

    return rows


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
    gold_data = load_gold_data(args.gold_file)

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

    feature_ids = {item["id"] for item in dev_dataset.features}
    gold_ids = {item["id"] for item in gold_data}

    if feature_ids != gold_ids:
        raise ValueError(
            "Dev feature IDs and gold IDs do not match"
        )

    model = ExtractiveQAModel(args.model_name).to(device)

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

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        return max(
            0.0,
            (total_steps - step) / max(
                1, total_steps - warmup_steps
            ),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"A_phobert_qa_lr{args.lr_label}_"
        f"bs{args.batch_size}_ga{args.gradient_accumulation_steps}_"
        f"seed{args.seed}"
    )

    run = init_run(
        name=run_name,
        job_type="fine_tune",
        config={
            "role": "A",
            "model": args.model_name,
            "architecture": "AutoModel + Linear(hidden_size, 2)",
            "loss": "average_start_end_cross_entropy",
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": 256,
            "doc_stride": 128,
            "seed": args.seed,
            "optimizer": "AdamW",
            "mixed_precision": use_amp,
            "gpu": torch.cuda.get_device_name(0) if use_amp else "CPU",
            "train_features": len(train_dataset),
            "dev_features": len(dev_dataset),
            "dev_questions": len(gold_data),
            "total_steps": total_steps,
            "early_stopping_patience": args.patience,
            "null_threshold": args.null_threshold,
        },
    )

    best_dev_f1 = float("-inf")
    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    global_step = 0
    total_start = time.perf_counter()

    try:
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)

            total_train_loss = 0.0
            total_train_samples = 0

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    start_logits, end_logits = model(
                        input_ids,
                        attention_mask,
                    )

                    loss = compute_loss(
                        start_logits,
                        end_logits,
                        start_positions,
                        end_positions,
                    )

                batch_size = input_ids.size(0)
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

                    current_lr = optimizer.param_groups[0]["lr"]

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1

                    run.log({
                        "train_step_loss": loss.item(),
                        "learning_rate": current_lr,
                        "global_step": global_step,
                    })

                if (step + 1) % 200 == 0:
                    print(
                        f"Epoch {epoch + 1}/{args.epochs} "
                        f"| Step {step + 1}/{len(train_loader)} "
                        f"| Loss {loss.item():.4f}"
                    )

            train_loss = total_train_loss / total_train_samples

            dev_metrics = evaluate(
                model,
                dev_loader,
                device,
                tokenizer,
                gold_data,
                args.max_answer_length,
                args.null_threshold,
                use_amp,
            )

            epoch_time = time.perf_counter() - epoch_start
            total_elapsed = time.perf_counter() - total_start

            dev_f1 = dev_metrics["all_f1"]

            improved = dev_f1 > best_dev_f1

            if improved:
                best_dev_f1 = dev_f1
                best_dev_loss = dev_metrics["dev_loss"]
                epochs_without_improvement = 0

                best_dir = output_dir / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)

                model.backbone.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

                torch.save(
                    model.qa_outputs.state_dict(),
                    best_dir / "qa_head.pt",
                )
                torch.save(
                    model.dropout.state_dict(),
                    best_dir / "qa_dropout.pt",
                )

                with (best_dir / "qa_config.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(
                        {
                            "hidden_size": model.backbone.config.hidden_size,
                            "max_answer_length": args.max_answer_length,
                            "null_threshold": args.null_threshold,
                        },
                        file,
                        indent=2,
                    )

                print(f"Saved best model: {best_dir}")
            else:
                epochs_without_improvement += 1

            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **dev_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": epoch_time,
                "total_training_time_seconds": total_elapsed,
                "best_dev_f1": best_dev_f1,
            }

            history.append(epoch_result)

            with (output_dir / "history.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(history, file, ensure_ascii=False, indent=2)

            run.log(epoch_result)

            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Dev loss: {dev_metrics['dev_loss']:.4f}")
            print(f"Dev EM: {dev_metrics['all_em']:.4f}")
            print(f"Dev F1: {dev_metrics['all_f1']:.4f}")
            print(f"Answerable EM: {dev_metrics['answerable_em']:.4f}")
            print(f"Answerable F1: {dev_metrics['answerable_f1']:.4f}")
            print(f"Impossible EM: {dev_metrics['impossible_em']:.4f}")
            print(f"Impossible F1: {dev_metrics['impossible_f1']:.4f}")
            print(f"Epoch time: {epoch_time / 60:.2f} minutes")

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {args.patience} "
                    "epochs without dev F1 improvement."
                )
                break

        total_time = time.perf_counter() - total_start

        run.summary["best_dev_f1"] = best_dev_f1
        run.summary["best_dev_loss"] = best_dev_loss
        run.summary["total_training_time_seconds"] = total_time
        run.summary["total_training_time_minutes"] = total_time / 60

        print(f"\nBest Dev F1: {best_dev_f1:.4f}")
        print(f"Best Dev Loss: {best_dev_loss:.4f}")
        print(f"Total training time: {total_time / 60:.2f} minutes")
        print(f"Training completed. Output: {output_dir}")

    finally:
        run.finish()


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
        "--gold_file",
        type=str,
        default="data/processed/viquad/dev.jsonl",
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
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--lr_label", type=str, default="3e-5")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--null_threshold", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
