# -*- coding: utf-8 -*-
"""Train/evaluate NSMC sentiment accuracy and save comparison plots.

This script is intentionally standalone so it can run from local Git Bash or
Colab after `python download_data.py` has prepared the data files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


RUN_CONFIGS = {
    "LIGHT": {
        "vocab_size": 2000,
        "context_length": 64,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 2,
        "batch_size": 8,
        "sentiment_train_limit": 3_000,
        "sentiment_val_limit": 1_000,
        "sentiment_test_limit": 1_000,
        "finetune_epochs": 2,
    },
    "BASIC": {
        "vocab_size": 3000,
        "context_length": 128,
        "emb_dim": 192,
        "n_heads": 4,
        "n_layers": 4,
        "batch_size": 8,
        "sentiment_train_limit": 10_000,
        "sentiment_val_limit": 2_000,
        "sentiment_test_limit": 2_000,
        "finetune_epochs": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune mini GPT on NSMC sentiment data and plot loss/accuracy.",
    )
    parser.add_argument("--run-level", choices=sorted(RUN_CONFIGS), default="BASIC")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="GPT checkpoint path, or 'auto'.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/sentiment_accuracy_metrics.jsonl"))
    parser.add_argument("--summary-path", type=Path, default=Path("logs/sentiment_accuracy_summary.json"))
    parser.add_argument("--plot-path", type=Path, default=Path("figures/sentiment_accuracy.png"))
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1, help="Refresh terminal progress every N batches.")
    parser.add_argument("--progress-width", type=int, default=28, help="Character width of the terminal progress bar.")
    parser.add_argument("--no-progress", action="store_true", help="Disable carriage-return batch progress.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/sentiment"))
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save a checkpoint every N epochs. 0 disables periodic checkpoints.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional checkpoint run directory name.")
    return parser.parse_args()


def add_src_to_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def resolve_output_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def find_latest_checkpoint(repo_root: Path) -> Path | None:
    ckpt_dir = repo_root / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def load_tokenizer(repo_root: Path, requested_vocab_size: int, tokenizer_path: Path | None):
    from bpe import BPETokenizer

    if tokenizer_path is not None:
        candidates = [resolve_output_path(repo_root, tokenizer_path)]
    else:
        candidates = [
            repo_root / "data" / f"vocab_bpe_{requested_vocab_size}.json",
            repo_root / "data" / "tokenizer.json",
        ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        tokenizer = BPETokenizer(vocab_size=requested_vocab_size)
        tokenizer.load(candidate)
        actual_vocab_size = len(tokenizer.id_to_token)
        print(f"tokenizer: {candidate} (vocab_size={actual_vocab_size})")
        return tokenizer, candidate, actual_vocab_size

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No tokenizer JSON found. Searched: {searched}")


def resolve_device(torch_module, requested: str, require_cuda: bool):
    if requested == "auto":
        device_name = "cuda" if torch_module.cuda.is_available() else "cpu"
    else:
        device_name = requested

    if device_name == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if require_cuda and device_name != "cuda":
        raise RuntimeError("CUDA is required, but no CUDA device is available.")

    device = torch_module.device(device_name)
    if device.type == "cuda":
        torch_module.backends.cuda.matmul.allow_tf32 = True
        print(f"device: cuda ({torch_module.cuda.get_device_name(0)})")
    else:
        print("device: cpu")
    return device


def load_backbone_checkpoint(model, checkpoint_arg: str | None, repo_root: Path, device) -> Path | None:
    if checkpoint_arg is None:
        print("checkpoint: none (training classifier from the current GPT backbone)")
        return None

    if checkpoint_arg == "auto":
        checkpoint_path = find_latest_checkpoint(repo_root)
        if checkpoint_path is None:
            print("checkpoint: auto requested, but no checkpoints/*.pt file was found")
            return None
    else:
        checkpoint_path = resolve_output_path(repo_root, Path(checkpoint_arg))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"checkpoint: loaded {checkpoint_path}")
    return checkpoint_path


def make_gpt_config(config: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "context_length": config["context_length"],
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "drop_rate": 0.1,
        "qkv_bias": False,
    }


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text.strip())
    return text.strip("_") or "sentiment_run"


def make_run_dir(
    repo_root: Path,
    checkpoint_dir: Path,
    run_name: str | None,
    run_level: str,
    config: dict[str, Any],
    train_limit: int,
    val_limit: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
) -> Path:
    root = resolve_output_path(repo_root, checkpoint_dir)
    if run_name is None:
        settings = (
            f"{run_level}_ctx{config['context_length']}_emb{config['emb_dim']}"
            f"_L{config['n_layers']}_H{config['n_heads']}_bs{batch_size}"
            f"_lr{lr:g}_train{train_limit}_val{val_limit}_ep{epochs}_seed{seed}"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{settings}_{timestamp}"
    run_dir = root / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    tokenizer_path: Path,
    loaded_checkpoint: Path | None,
    gpt_config: dict[str, Any],
    limits: dict[str, int],
    batch_size: int,
    epochs: int,
) -> Path:
    config_path = run_dir / "run_config.json"
    payload = {
        "run_level": args.run_level,
        "device": args.device,
        "require_cuda": args.require_cuda,
        "tokenizer_path": str(tokenizer_path),
        "loaded_checkpoint": str(loaded_checkpoint) if loaded_checkpoint is not None else None,
        "gpt_config": gpt_config,
        "limits": limits,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def save_sentiment_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    epoch: int,
    history: list[dict[str, float]],
    test_metrics: dict[str, float] | None,
    gpt_config: dict[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "test": test_metrics,
            "gpt_config": gpt_config,
        },
        path,
    )


def print_progress(
    enabled: bool,
    message: str,
    *,
    end: bool = False,
    last_len: list[int],
) -> None:
    if not enabled:
        return
    if end:
        sys.stdout.write("\r" + message + " " * max(0, last_len[0] - len(message)) + "\n")
        last_len[0] = 0
    else:
        sys.stdout.write("\r" + message + " " * max(0, last_len[0] - len(message)))
        last_len[0] = len(message)
    sys.stdout.flush()


def format_progress_bar(current: int, total: int, width: int) -> str:
    width = max(10, width)
    if total <= 0:
        return "[" + "-" * width + "]   0.0%"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}%"


def train_epoch_with_progress(
    model,
    train_loader,
    optimizer,
    device,
    *,
    epoch: int,
    total_epochs: int,
    progress: bool,
    progress_every: int,
    progress_width: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)
    line_len = [0]

    for batch_idx, (input_ids, labels) in enumerate(train_loader, start=1):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += batch_size

        if batch_idx % progress_every == 0 or batch_idx == total_batches:
            avg_loss = total_loss / total if total else float("nan")
            avg_acc = correct / total if total else 0.0
            print_progress(
                progress,
                (
                    f"epoch {epoch:03d}/{total_epochs:03d} "
                    f"train {format_progress_bar(batch_idx, total_batches, progress_width)} "
                    f"batch {batch_idx:04d}/{total_batches:04d} "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
                ),
                last_len=line_len,
            )

    avg_loss = total_loss / total if total else float("nan")
    accuracy = correct / total if total else 0.0
    print_progress(
        progress,
        f"epoch {epoch:03d}/{total_epochs:03d} train done loss={avg_loss:.4f} acc={accuracy:.4f}",
        end=True,
        last_len=line_len,
    )
    return avg_loss, accuracy


def evaluate_with_progress(
    model,
    data_loader,
    device,
    *,
    split: str,
    epoch: int,
    total_epochs: int,
    progress: bool,
    progress_every: int,
    progress_width: int,
) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(data_loader)
    line_len = [0]

    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(data_loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss, logits = model(input_ids, labels=labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += batch_size

            if batch_idx % progress_every == 0 or batch_idx == total_batches:
                avg_loss = total_loss / total if total else float("nan")
                avg_acc = correct / total if total else 0.0
                print_progress(
                    progress,
                    (
                        f"epoch {epoch:03d}/{total_epochs:03d} "
                        f"{split} {format_progress_bar(batch_idx, total_batches, progress_width)} "
                        f"batch {batch_idx:04d}/{total_batches:04d} "
                        f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
                    ),
                    last_len=line_len,
                )

    avg_loss = total_loss / total if total else float("nan")
    accuracy = correct / total if total else 0.0
    print_progress(
        progress,
        f"epoch {epoch:03d}/{total_epochs:03d} {split} done loss={avg_loss:.4f} acc={accuracy:.4f}",
        end=True,
        last_len=line_len,
    )
    return avg_loss, accuracy


def plot_history(history: list[dict[str, float]], test_metrics: dict[str, float], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    loss_ax, acc_ax = axes

    loss_ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    if test_metrics:
        loss_ax.axhline(test_metrics["loss"], linestyle="--", color="tab:gray", label="test")
    loss_ax.set_title("Sentiment loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross entropy")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()

    acc_ax.plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    acc_ax.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    if test_metrics:
        acc_ax.axhline(test_metrics["accuracy"], linestyle="--", color="tab:gray", label="test")
    acc_ax.set_title("Sentiment accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_ylim(0.0, 1.0)
    acc_ax.grid(alpha=0.25)
    acc_ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    add_src_to_path(repo_root)

    import torch
    from torch.utils.data import DataLoader

    from finetune import (
        GPTForSequenceClassification,
        ReviewSentimentDataset,
    )
    from metrics import append_jsonl_record
    from model import GPTModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = dict(RUN_CONFIGS[args.run_level])
    tokenizer, tokenizer_path, actual_vocab_size = load_tokenizer(
        repo_root,
        requested_vocab_size=config["vocab_size"],
        tokenizer_path=args.tokenizer_path,
    )
    config["vocab_size"] = actual_vocab_size

    train_limit = args.train_limit if args.train_limit is not None else config["sentiment_train_limit"]
    val_limit = args.val_limit if args.val_limit is not None else config["sentiment_val_limit"]
    test_limit = args.test_limit if args.test_limit is not None else config["sentiment_test_limit"]
    epochs = args.epochs if args.epochs is not None else config["finetune_epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    progress = not args.no_progress
    progress_every = max(1, args.progress_every)
    progress_width = max(10, args.progress_width)
    checkpoint_every = max(0, args.checkpoint_every)
    limits = {"train": train_limit, "val": val_limit, "test": test_limit}

    train_data = read_jsonl(repo_root / "data" / "nsmc_sentiment_train.jsonl", train_limit)
    val_data = read_jsonl(repo_root / "data" / "nsmc_sentiment_val.jsonl", val_limit)
    test_data = read_jsonl(repo_root / "data" / "nsmc_sentiment_test.jsonl", test_limit)
    print(f"examples: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    device = resolve_device(torch, args.device, args.require_cuda)
    gpt_config = make_gpt_config(config, actual_vocab_size)
    backbone = GPTModel(gpt_config)
    loaded_checkpoint = load_backbone_checkpoint(backbone, args.checkpoint, repo_root, device)
    clf_model = GPTForSequenceClassification(backbone, num_labels=2).to(device)
    run_dir = make_run_dir(
        repo_root,
        args.checkpoint_dir,
        args.run_name,
        args.run_level,
        config,
        train_limit,
        val_limit,
        batch_size,
        epochs,
        args.lr,
        args.seed,
    )
    run_config_path = write_run_config(
        run_dir,
        args=args,
        tokenizer_path=tokenizer_path,
        loaded_checkpoint=loaded_checkpoint,
        gpt_config=gpt_config,
        limits=limits,
        batch_size=batch_size,
        epochs=epochs,
    )
    print(f"checkpoint run dir: {run_dir}")
    print(f"run config: {run_config_path}")

    train_ds = ReviewSentimentDataset(train_data, tokenizer, max_length=config["context_length"])
    val_ds = ReviewSentimentDataset(val_data, tokenizer, max_length=config["context_length"])
    test_ds = ReviewSentimentDataset(test_data, tokenizer, max_length=config["context_length"])

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    metrics_path = resolve_output_path(repo_root, args.metrics_path)
    summary_path = resolve_output_path(repo_root, args.summary_path)
    plot_path = resolve_output_path(repo_root, args.plot_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists() and not args.append_metrics:
        metrics_path.unlink()

    optimizer = torch.optim.AdamW(clf_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_checkpoint_path = run_dir / "best.pt"
    final_checkpoint_path = run_dir / "final.pt"
    test_metrics: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch_with_progress(
            clf_model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=epochs,
            progress=progress,
            progress_every=progress_every,
            progress_width=progress_width,
        )
        append_jsonl_record(
            metrics_path,
            {
                "stage": "sentiment",
                "event": "train_epoch",
                "split": "train",
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_acc,
                "num_examples": len(train_data),
            },
        )
        val_loss, val_acc = evaluate_with_progress(
            clf_model,
            val_loader,
            device,
            split="val",
            epoch=epoch,
            total_epochs=epochs,
            progress=progress,
            progress_every=progress_every,
            progress_width=progress_width,
        )
        append_jsonl_record(
            metrics_path,
            {
                "stage": "sentiment",
                "event": "evaluate",
                "split": "val",
                "epoch": epoch,
                "loss": val_loss,
                "accuracy": val_acc,
                "num_examples": len(val_data),
            },
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"epoch {epoch}: "
            f"train loss={train_loss:.4f}, train acc={train_acc:.4f}, "
            f"val loss={val_loss:.4f}, val acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_sentiment_checkpoint(
                best_checkpoint_path,
                clf_model,
                optimizer,
                epoch=epoch,
                history=history,
                test_metrics=test_metrics,
                gpt_config=gpt_config,
            )
            print(f"best checkpoint saved: {best_checkpoint_path}")

        if checkpoint_every and epoch % checkpoint_every == 0:
            epoch_checkpoint_path = run_dir / f"epoch_{epoch:03d}.pt"
            save_sentiment_checkpoint(
                epoch_checkpoint_path,
                clf_model,
                optimizer,
                epoch=epoch,
                history=history,
                test_metrics=test_metrics,
                gpt_config=gpt_config,
            )
            print(f"epoch checkpoint saved: {epoch_checkpoint_path}")

    test_loss, test_acc = evaluate_with_progress(
        clf_model,
        test_loader,
        device,
        split="test",
        epoch=epochs,
        total_epochs=epochs,
        progress=progress,
        progress_every=progress_every,
        progress_width=progress_width,
    )
    test_metrics = {"loss": test_loss, "accuracy": test_acc}
    append_jsonl_record(
        metrics_path,
        {
            "stage": "sentiment",
            "event": "evaluate",
            "split": "test",
            "epoch": epochs,
            "loss": test_loss,
            "accuracy": test_acc,
            "num_examples": len(test_data),
        },
    )
    print(f"test: loss={test_loss:.4f}, acc={test_acc:.4f}")
    save_sentiment_checkpoint(
        final_checkpoint_path,
        clf_model,
        optimizer,
        epoch=epochs,
        history=history,
        test_metrics=test_metrics,
        gpt_config=gpt_config,
    )
    print(f"final checkpoint saved: {final_checkpoint_path}")

    plot_history(history, test_metrics, plot_path)

    summary = {
        "run_level": args.run_level,
        "device": str(device),
        "tokenizer_path": str(tokenizer_path),
        "loaded_checkpoint_path": str(loaded_checkpoint) if loaded_checkpoint is not None else None,
        "checkpoint_run_dir": str(run_dir),
        "run_config_path": str(run_config_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "final_checkpoint_path": str(final_checkpoint_path),
        "config": gpt_config,
        "limits": limits,
        "batch_size": batch_size,
        "epochs": epochs,
        "history": history,
        "test": test_metrics,
        "metrics_path": str(metrics_path),
        "plot_path": str(plot_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    run()
