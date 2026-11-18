"""Khởi tạo run W&B dùng chung cho cả nhóm.

Quy ước (docs/project_plan.md): tên run theo mẫu `<role>_<model>_<cấu hình>_seed<k>`,
kèm commit hash. Entity và project mặc định là của nhóm; ghi đè bằng biến môi trường
WANDB_ENTITY / WANDB_PROJECT. Đặt WANDB_MODE=disabled (ví dụ trong CI) để tắt hẳn W&B.

Ví dụ:
    from src.tracking import init_run
    run = init_run("B_bge-m3_index-domain_win200s100", job_type="build_index",
                   config={"max_words": 200, "stride": 100})
    run.summary["n_chunks"] = 209
    run.finish()
"""
import os
import subprocess
from typing import Any, Dict, Optional

DEFAULT_ENTITY = "nhom09-vi-qa-rag"
DEFAULT_PROJECT = "vi-qa-rag"


def git_commit_hash() -> str:
    """Commit hash ngắn của HEAD; 'unknown' nếu không chạy trong repo git (ví dụ Kaggle)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def init_run(name: str, job_type: str, config: Optional[Dict[str, Any]] = None):
    """Gọi wandb.init với entity/project của nhóm và tự gắn commit hash vào config."""
    import wandb

    cfg = dict(config or {})
    cfg.setdefault("commit", git_commit_hash())
    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
        project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        name=name,
        job_type=job_type,
        config=cfg,
    )
