# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import csv
import json
import math
from pathlib import Path

import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.0
    batches_seen = 0
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    model.eval()
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            if batches_seen >= max_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            batches_seen += 1

    model.train()
    return total_loss / batches_seen


def calc_metrics_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> dict[str, float]:
    """data_loader의 평균 loss, token accuracy, perplexity를 계산합니다."""
    if data_loader is None or len(data_loader) == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "perplexity": float("nan"),
        }

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches_seen = 0
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    model.eval()
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            if batches_seen >= max_batches:
                break

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            loss, logits = model(input_batch, targets=target_batch)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == target_batch).sum().item()
            total_tokens += target_batch.numel()
            batches_seen += 1

    model.train()
    avg_loss = total_loss / max(1, batches_seen)
    return {
        "loss": avg_loss,
        "accuracy": total_correct / max(1, total_tokens),
        "perplexity": math.exp(avg_loss) if avg_loss < 100 else float("inf"),
    }


def save_metrics_history(history: list[dict], path: str | Path) -> None:
    """실험 history를 그래프용 JSONL 또는 CSV 파일로 저장합니다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        fieldnames = sorted({key for row in history for key in row.keys()})
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        return

    with output_path.open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
    metrics_history: list[dict] | None = None,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics_history": metrics_history or [],
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_top_value = top_values[:, [-1]]
            logits = torch.where(
                logits < min_top_value,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if temperature <= 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and torch.all(idx_next == eos_id):
            idx = torch.cat((idx, idx_next), dim=1)
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    model.eval()
    encoded = tokenizer.encode(start_context, add_bos_eos=False)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    token_ids = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None,
    )
    print(tokenizer.decode(token_ids[0].tolist(), skip_special=True))


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    test_loader=None,
    ckpt_freq: int | None = None,
    metrics_path: str | Path | None = None,
    return_history: bool = False,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float] | dict[str, list]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    train_losses = []
    val_losses = []
    test_losses = []
    history = []

    model.to(device)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0.0
        batches_seen = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches_seen += 1
            global_step += 1

            if eval_freq and global_step % eval_freq == 0 and val_loader is not None:
                train_loss = total_loss / max(1, batches_seen)
                val_metrics = calc_metrics_loader(val_loader, model, device, num_batches=eval_iter)
                row = {
                    "phase": "eval",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_perplexity": val_metrics["perplexity"],
                }
                history.append(row)
                val_losses.append(val_metrics["loss"])
                print(
                    f"step {global_step}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}, "
                    f"val_ppl={val_metrics['perplexity']:.2f}"
                )
                if metrics_path is not None:
                    save_metrics_history(history, metrics_path)

            if ckpt_freq and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    f"checkpoint_step_{global_step}.pt",
                    metrics_history=history,
                )

        epoch_train_loss = total_loss / max(1, batches_seen)
        train_losses.append(epoch_train_loss)
        row = {
            "phase": "epoch",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_train_loss,
        }

        if val_loader is not None:
            val_metrics = calc_metrics_loader(val_loader, model, device, num_batches=eval_iter)
            row.update({
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_perplexity": val_metrics["perplexity"],
            })
            val_losses.append(val_metrics["loss"])

        if test_loader is not None:
            test_metrics = calc_metrics_loader(test_loader, model, device, num_batches=eval_iter)
            row.update({
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_perplexity": test_metrics["perplexity"],
            })
            test_losses.append(test_metrics["loss"])

        history.append(row)
        print(
            f"epoch {epoch}: "
            f"train_loss={epoch_train_loss:.4f}"
            + (f", val_loss={row['val_loss']:.4f}" if "val_loss" in row else "")
            + (f", test_loss={row['test_loss']:.4f}" if "test_loss" in row else "")
        )

        if metrics_path is not None:
            save_metrics_history(history, metrics_path)

        if tokenizer is not None and start_context:
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context,
                context_size=model.config.get("context_length", 256),
            )

    if return_history:
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_losses": test_losses,
            "history": history,
        }

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    import matplotlib.pyplot as plt

    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
