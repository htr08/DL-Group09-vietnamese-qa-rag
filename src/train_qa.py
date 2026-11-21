
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
from src.qa_metrics import exact_match_score, f1_score
from src.qa_nbest import decode_window
from src.qa_resume import save_training_state, load_training_state
# QA-REVIEW-STAGE2: unified offset decoder + ignored-label loss


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
    """Gộp batch; giữ offset metadata cho decoder chính xác."""
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask = [], []
    start_positions, end_positions = [], []
    ids, impossible = [], []
    word_spans, subword_spans = [], []
    for item in batch:
        ids_list = item["input_ids"]
        padding_length = max_length - len(ids_list)
        input_ids.append(ids_list + [pad_token_id] * padding_length)
        attention_mask.append([1] * len(ids_list) + [0] * padding_length)
        start_positions.append(item["start_positions"])
        end_positions.append(item["end_positions"])
        ids.append(item["id"])
        impossible.append(item["is_impossible"])
        word_spans.append(item.get("word_spans"))
        subword_spans.append(item.get("subword_spans"))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "start_positions": torch.tensor(start_positions, dtype=torch.long),
        "end_positions": torch.tensor(end_positions, dtype=torch.long),
        "ids": ids, "is_impossible": impossible,
        "word_spans": word_spans, "subword_spans": subword_spans,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(start_logits, end_logits, start_positions, end_positions):
    """Trung bình loss trên các feature có nhãn; -100 được bỏ qua.

    Nếu toàn bộ batch không có nhãn hợp lệ, trả zero-gradient loss;
    evaluator không cộng batch ấy vào Dev loss nhưng vẫn chấm EM/F1.
    """
    valid = (start_positions != -100) & (end_positions != -100)
    start = nn.functional.cross_entropy(
        start_logits, start_positions, ignore_index=-100, reduction="none"
    )
    end = nn.functional.cross_entropy(
        end_logits, end_positions, ignore_index=-100, reduction="none"
    )
    if not bool(valid.any()):
        return (start_logits.sum() + end_logits.sum()) * 0.0
    return ((start + end) * 0.5)[valid].mean()


def decode_feature(input_ids, attention_mask, start_logits, end_logits,
                   tokenizer, max_answer_length):
    """Legacy wrapper; cả training và standalone đều dùng decode_window.

    Không dùng hàm này để báo cáo điểm Dev mới: thiếu metadata offset.
    """
    valid_length = int(attention_mask.sum().item())
    ids = input_ids[:valid_length].detach().cpu().tolist()
    start = start_logits[:valid_length].float().detach().cpu().tolist()
    end = end_logits[:valid_length].float().detach().cpu().tolist()
    candidates, null_score = decode_window(
        ids, start, end, tokenizer, n_best_size=max(20, valid_length),
        max_answer_length=max_answer_length,
        sep_token_id=tokenizer.sep_token_id,
    )
    best = candidates[0] if candidates else None
    return (best["text"] if best else "", null_score,
            best["score"] if best else float("-inf"))


def evaluate(model, loader, device, tokenizer, gold_data,
             max_answer_length, null_threshold, use_amp):
    """Chấm EM/F1 toàn bộ Dev bằng cùng decoder offset với standalone.

    Gold chỉ dùng để lấy *context gốc* và references, không làm thay đổi
    input_ids/cửa sổ. Dev loss bỏ qua label -100; EM/F1 vẫn chấm mọi câu.
    """
    model.eval()
    total_loss = 0.0
    total_labeled = 0
    predictions = defaultdict(lambda: {
        "best_answer": "", "best_score": float("-inf"),
        "null_score": float("inf"),
    })
    gold_by_id = {str(r["id"]): r for r in gold_data}
    if len(gold_by_id) != len(gold_data):
        raise ValueError("Duplicate gold IDs")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=use_amp):
                start_logits, end_logits = model(input_ids, attention_mask)
                loss = compute_loss(start_logits, end_logits,
                                    start_positions, end_positions)
            valid = (start_positions != -100) & (end_positions != -100)
            valid_count = int(valid.sum().item())
            if valid_count:
                total_loss += float(loss.item()) * valid_count
                total_labeled += valid_count
            # Mỗi batch chỉ chuyển GPU -> CPU một lần, tránh .item() trong vòng đôi.
            starts = start_logits.float().cpu().tolist()
            ends = end_logits.float().cpu().tolist()
            ids_cpu = input_ids.cpu().tolist()
            lengths = attention_mask.sum(dim=1).cpu().tolist()
            for i, qid in enumerate(batch["ids"]):
                question_id = str(qid)
                if question_id not in gold_by_id:
                    raise ValueError(f"Missing gold for {question_id}")
                word_spans = batch["word_spans"][i]
                subword_spans = batch["subword_spans"][i]
                if word_spans is None or subword_spans is None:
                    raise ValueError(
                        "Dev feature missing offsets: preprocess again in eval mode"
                    )
                length = int(lengths[i])
                candidates, null_score = decode_window(
                    ids_cpu[i][:length], starts[i][:length], ends[i][:length],
                    tokenizer, n_best_size=20, max_answer_length=max_answer_length,
                    sep_token_id=tokenizer.sep_token_id,
                    word_spans=word_spans, subword_spans=subword_spans,
                    context=gold_by_id[question_id]["context"],
                )
                record = predictions[question_id]
                record["null_score"] = min(record["null_score"], null_score)
                if candidates and candidates[0]["score"] > record["best_score"]:
                    record["best_score"] = candidates[0]["score"]
                    record["best_answer"] = candidates[0]["text"]

    score_groups = {
        "all": {"em": [], "f1": []},
        "answerable": {"em": [], "f1": []},
        "impossible": {"em": [], "f1": []},
    }
    for row in gold_data:
        qid = str(row["id"])
        item = predictions.get(qid)
        if item is None:
            raise ValueError(f"Missing prediction for question {qid}")
        score_diff = item["null_score"] - item["best_score"]
        prediction = "" if score_diff > null_threshold else item["best_answer"]
        refs = [""] if row["is_impossible"] else row["answers"]["text"]
        cat = "impossible" if row["is_impossible"] else "answerable"
        em = float(exact_match_score(prediction, refs))
        f1 = float(f1_score(prediction, refs))
        score_groups["all"]["em"].append(em)
        score_groups["all"]["f1"].append(f1)
        score_groups[cat]["em"].append(em)
        score_groups[cat]["f1"].append(f1)
    result = {"dev_loss": total_loss / total_labeled if total_labeled else None}
    for cat, group in score_groups.items():
        for metric, values in group.items():
            result[f"{cat}_{metric}"] = float(np.mean(values)) if values else 0.0
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

    model_tag = (
        args.model_name.replace("\\", "/").rstrip("/").split("/")[-1]
        .replace("-", "_")
    )
    run_name = (
        f"A_{model_tag}_qa_lr{args.lr_label}_"
        f"bs{args.batch_size}_ga{args.gradient_accumulation_steps}_"
        f"seed{args.seed}"
    )

    # Đọc thông số thực tế từ report, thay vì hard-code vào W&B.
    # Report cũ (từ lần train đã hoàn thành) có thể chưa có metadata.
    def _read_feature_report_config(feature_file, split):
        import json

        report_path = Path(feature_file).resolve().with_name(f"{split}_report.json")
        if not report_path.is_file():
            return None
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        config = payload.get("preprocessing_config")
        if config is None:
            return None
        if not isinstance(config, dict):
            raise ValueError(f"Invalid preprocessing_config: {report_path}")
        result = {
            "max_length": config.get("max_length"),
            "doc_stride": config.get("doc_stride"),
        }
        max_length, stride = result.values()
        if (type(max_length) is not int or type(stride) is not int
                or max_length <= 0 or stride <= 0):
            raise ValueError(f"Invalid preprocessing parameters: {report_path}")
        return result

    train_pre_cfg = _read_feature_report_config(args.train_file, "train")
    dev_pre_cfg = _read_feature_report_config(args.dev_file, "dev")
    if train_pre_cfg is not None and dev_pre_cfg is not None:
        if train_pre_cfg != dev_pre_cfg:
            raise ValueError("Train/Dev preprocessing config mismatch")
        preprocessing_meta = dict(train_pre_cfg)
        preprocessing_meta["source"] = "train_and_dev_feature_reports"
    elif train_pre_cfg is not None or dev_pre_cfg is not None:
        raise ValueError(
            "Only one Train/Dev report has preprocessing_config; "
            "regenerate both reports consistently before training"
        )
    else:
        # Tương thích report lịch sử: dùng constants, đánh dấu CHƯA XÁC MINH
        # theo report; không sửa hoặc viết đè các report cũ.
        from src.preprocessing_qa import MAX_LEN, DOC_STRIDE
        preprocessing_meta = {
            "max_length": MAX_LEN,
            "doc_stride": DOC_STRIDE,
            "source": "legacy_constants_unverified_by_feature_reports",
        }
        print("WARNING: Train/Dev reports have no preprocessing_config. "
              "W&B parameters use current source constants, not verified "
              "historical report values.")

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
            "max_length": preprocessing_meta["max_length"],
            "doc_stride": preprocessing_meta["doc_stride"],
            "preprocessing_config_source": preprocessing_meta["source"],
            "seed": args.seed,
            "optimizer": "AdamW",
            "mixed_precision": use_amp,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else "CPU"
            ),
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

    # Resume chỉ tại ranh giới epoch. Cấu hình phải trùng lần chạy gốc.
    resume_config = {
        "model_name": args.model_name,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "null_threshold": float(args.null_threshold),
        "train_features": len(train_dataset),
        "dev_features": len(dev_dataset),
    }
    start_epoch = 0
    resume_path = getattr(args, "resume_checkpoint", None)
    if resume_path:
        state = load_training_state(
            resume_path, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            expected_config=resume_config,
        )
        start_epoch = state["completed_epoch"]
        if start_epoch >= args.epochs:
            raise ValueError("Checkpoint đã hoàn tất số epochs cấu hình")
        if not (output_dir / "best_model" / "qa_head.pt").is_file():
            raise FileNotFoundError(
                "Resume cần best_model trong --output_dir để giữ best checkpoint "
                "của các epoch trước; hãy dùng lại đúng output_dir"
            )
        best_dev_f1 = state["best_dev_f1"]
        best_dev_loss = state["best_dev_loss"]
        epochs_without_improvement = state["epochs_without_improvement"]
        global_step = state["global_step"]
        history = state["history"]
        print(f"Resumed epoch {start_epoch}/{args.epochs} from {resume_path}")


    try:
        for epoch in range(start_epoch, args.epochs):
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
                    loss / min(args.gradient_accumulation_steps, len(train_loader) - (step // args.gradient_accumulation_steps) * args.gradient_accumulation_steps)
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

            # Ghi checkpoint đầy đủ mỗi epoch (atomic), ngoài best_model.
            save_training_state(
                output_dir / "last_checkpoint.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, completed_epoch=epoch + 1,
                best_dev_f1=best_dev_f1, best_dev_loss=best_dev_loss,
                epochs_without_improvement=epochs_without_improvement,
                global_step=global_step, history=history,
                config=resume_config,
            )


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
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Tiếp tục training từ last_checkpoint.pt cùng output_dir")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
