"""Lưu/khôi phục checkpoint đầy đủ tại ranh giới epoch của extractive QA.

Chỉ tải checkpoint do chính dự án tạo. torch.load(weights_only=False) có thể
thực thi pickle không an toàn nếu file đến từ nguồn không tin cậy.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


STATE_VERSION = 1


def save_training_state(path, *, model, optimizer, scheduler, scaler,
                        completed_epoch, best_dev_f1, best_dev_loss,
                        epochs_without_improvement, global_step, history, config):
    """Lưu trạng thái có thể resume sau epoch; ghi atomic để tránh file dở."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": STATE_VERSION,
        "completed_epoch": int(completed_epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_dev_f1": float(best_dev_f1),
        "best_dev_loss": float(best_dev_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "global_step": int(global_step),
        "history": list(history),
        "config": dict(config),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_name(path.name + ".incomplete")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_state(path, *, model, optimizer, scheduler, scaler, expected_config):
    """Khôi phục model/optimizer/scheduler/scaler/RNG và trả tiến độ epoch."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Cần weights_only=False vì checkpoint lưu Python/NumPy RNG states.
    # Chỉ được load file do chính mình tin cậy và tạo ra.
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        raise ValueError("Checkpoint version không hợp lệ")
    actual = state.get("config")
    if actual != dict(expected_config):
        differences = {
            k: (actual.get(k) if isinstance(actual, dict) else None, v)
            for k, v in dict(expected_config).items()
            if not isinstance(actual, dict) or actual.get(k) != v
        }
        raise ValueError(f"Resume config khác cấu hình training hiện tại: {differences}")
    completed = state.get("completed_epoch")
    if type(completed) is not int or completed < 1:
        raise ValueError("Checkpoint thiếu completed_epoch hợp lệ")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"].cpu())
    if torch.cuda.is_available() and state.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return {
        "completed_epoch": completed,
        "best_dev_f1": float(state["best_dev_f1"]),
        "best_dev_loss": float(state["best_dev_loss"]),
        "epochs_without_improvement": int(state["epochs_without_improvement"]),
        "global_step": int(state["global_step"]),
        "history": list(state["history"]),
    }
