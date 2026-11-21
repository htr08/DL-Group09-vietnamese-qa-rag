"""Kiểm thử checkpoint resume trên CPU, không tải PhoBERT/checkpoint Kaggle."""

import random

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch",
    reason="CI tối giản không cài PyTorch",
)

from src.qa_resume import save_training_state, load_training_state


def make_components():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def test_epoch_resume_roundtrip(tmp_path):
    random.seed(40)
    np.random.seed(40)
    torch.manual_seed(40)
    model, optimizer, scheduler, scaler = make_components()
    # Một bước thật để khởi tạo state của AdamW.
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    saved_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    config = {"model_name": "dummy", "epochs": 3, "batch_size": 2}
    path = tmp_path / "last_checkpoint.pt"
    save_training_state(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, completed_epoch=1, best_dev_f1=0.6,
        best_dev_loss=1.2, epochs_without_improvement=0,
        global_step=1, history=[{"epoch": 1}], config=config,
    )
    expected_randoms = (random.random(), np.random.rand(), torch.rand(1).item())
    # Cố ý làm xáo trộn model và RNG trước khi restore.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(7)
    random.random()
    np.random.rand()
    torch.rand(1)
    state = load_training_state(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, expected_config=config,
    )
    assert state["completed_epoch"] == 1
    assert state["global_step"] == 1
    assert state["history"] == [{"epoch": 1}]
    assert all(torch.equal(model.state_dict()[k], v) for k, v in saved_weights.items())
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected_randoms
    assert optimizer.state_dict()["state"]
    assert scheduler.last_epoch >= 0
    assert not (tmp_path / "last_checkpoint.pt.incomplete").exists()


def test_resume_refuses_different_config_before_loading(tmp_path):
    model, optimizer, scheduler, scaler = make_components()
    path = tmp_path / "checkpoint.pt"
    save_training_state(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, completed_epoch=1, best_dev_f1=0.5,
        best_dev_loss=1.5, epochs_without_improvement=0,
        global_step=1, history=[], config={"epochs": 3},
    )
    with pytest.raises(ValueError, match="Resume config"):
        load_training_state(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, expected_config={"epochs": 5},
        )
