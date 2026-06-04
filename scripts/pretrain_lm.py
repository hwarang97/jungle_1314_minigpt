# -*- coding: utf-8 -*-
"""Pretrain the mini GPT language model from local NSMC LM text files."""

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
        "vocab_size": 3000,
        "context_length": 64,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
        "batch_size": 16,
        "train_chars": 500_000,
        "val_chars": 50_000,
        "epochs": 5,
    },
    "BASIC": {
        "vocab_size": 3000,
        "context_length": 128,
        "emb_dim": 192,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "batch_size": 32,
        "train_chars": 1_500_000,
        "val_chars": 200_000,
        "epochs": 20,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain BASIC-compatible mini GPT on nsmc_lm_train.txt.",
    )
    parser.add_argument("--run-level", choices=sorted(RUN_CONFIGS), default="BASIC")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--train-text-path", type=Path, default=Path("data/nsmc_lm_train.txt"))
    parser.add_argument("--val-text-path", type=Path, default=Path("data/nsmc_lm_val.txt"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None, help="Override the preset attention head count.")
    parser.add_argument("--train-chars", type=int, default=None)
    parser.add_argument("--val-chars", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-freq", type=int, default=0, help="Optional step-based eval frequency. 0 keeps epoch-only eval.")
    parser.add_argument("--eval-iter", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-context", type=str, default="이 영화는")
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--progress-width", type=int, default=28)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain"))
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save a checkpoint every N epochs. 0 disables periodic checkpoints.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after N epochs without validation loss improvement. 0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum validation loss improvement required to reset patience.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from, or 'auto'.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/pretrain_basic_metrics.jsonl"))
    parser.add_argument("--summary-path", type=Path, default=Path("logs/pretrain_basic_summary.json"))
    parser.add_argument("--plot-path", type=Path, default=Path("figures/pretrain_basic_loss.png"))
    parser.add_argument("--sample-path", type=Path, default=Path("logs/pretrain_basic_samples.jsonl"))
    parser.add_argument("--append-metrics", action="store_true")
    return parser.parse_args()


def add_src_to_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text.strip())
    return text.strip("_") or "pretrain_run"


def format_progress_bar(current: int, total: int, width: int) -> str:
    width = max(10, width)
    if total <= 0:
        return "[" + "-" * width + "]   0.0%"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio * 100:5.1f}%"


def print_progress(enabled: bool, message: str, *, last_len: list[int], end: bool = False) -> None:
    if not enabled:
        return
    if end:
        sys.stdout.write("\r" + message + " " * max(0, last_len[0] - len(message)) + "\n")
        last_len[0] = 0
    else:
        sys.stdout.write("\r" + message + " " * max(0, last_len[0] - len(message)))
        last_len[0] = len(message)
    sys.stdout.flush()


def make_gpt_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "vocab_size": config["vocab_size"],
        "context_length": config["context_length"],
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "drop_rate": config["drop_rate"],
        "qkv_bias": False,
    }


def make_run_dir(
    repo_root: Path,
    checkpoint_dir: Path,
    run_name: str | None,
    run_level: str,
    config: dict[str, Any],
    batch_size: int,
    epochs: int,
    lr: float,
    train_chars: int,
    val_chars: int,
    seed: int,
) -> Path:
    root = resolve_path(repo_root, checkpoint_dir)
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{run_level}_ctx{config['context_length']}_emb{config['emb_dim']}"
            f"_L{config['n_layers']}_H{config['n_heads']}_bs{batch_size}"
            f"_lr{lr:g}_chars{train_chars}_val{val_chars}_ep{epochs}_seed{seed}_{timestamp}"
        )
    run_dir = root / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_checkpoint(repo_root: Path) -> Path | None:
    ckpt_dir = repo_root / "checkpoints" / "pretrain"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.rglob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def save_pretrain_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    final_result: dict[str, Any] | None = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "config": config,
            "final_result": final_result,
        },
        path,
    )


def load_resume_checkpoint(model, optimizer, resume_arg: str | None, repo_root: Path, device) -> tuple[Path | None, int, int, list[dict[str, Any]]]:
    if resume_arg is None:
        return None, 1, 0, []
    if resume_arg == "auto":
        checkpoint_path = find_latest_checkpoint(repo_root)
        if checkpoint_path is None:
            print("resume: auto requested, but no pretrain checkpoint found")
            return None, 1, 0, []
    else:
        checkpoint_path = resolve_path(repo_root, Path(resume_arg))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    history = list(checkpoint.get("history", []))
    print(f"resume: {checkpoint_path} epoch={last_epoch}, global_step={global_step}")
    return checkpoint_path, last_epoch + 1, global_step, history


def load_tokenizer(repo_root: Path, tokenizer_path: Path, expected_vocab_size: int):
    from bpe import BPETokenizer

    resolved = resolve_path(repo_root, tokenizer_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Tokenizer JSON not found: {resolved}")
    tokenizer = BPETokenizer(vocab_size=expected_vocab_size)
    tokenizer.load(resolved)
    actual_vocab_size = len(tokenizer.id_to_token)
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(f"Tokenizer vocab_size mismatch: expected {expected_vocab_size}, got {actual_vocab_size}")
    print(f"tokenizer: {resolved} (vocab_size={actual_vocab_size})")
    return tokenizer, resolved


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


def plot_history(history: list[dict[str, Any]], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch_rows = [row for row in history if row.get("event") == "epoch"]
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    if epoch_rows:
        epochs = [row["epoch"] for row in epoch_rows]
        plt.plot(epochs, [row["train_loss"] for row in epoch_rows], marker="o", label="train loss")
        val_rows = [row for row in epoch_rows if row.get("val_loss") is not None]
        if val_rows:
            plt.plot([row["epoch"] for row in val_rows], [row["val_loss"] for row in val_rows], marker="o", label="validation loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mini GPT pretraining loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def generate_sample(model, tokenizer, device, config: dict[str, Any], context: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    import torch
    from train import generate

    encoded = tokenizer.encode(context, add_bos_eos=False)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id(),
    )
    return tokenizer.decode(generated_ids[0].tolist(), skip_special=True)


def run() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    add_src_to_path(repo_root)

    import torch

    from dataset import create_dataloader
    from metrics import append_jsonl_record
    from model import GPTModel
    from train import calc_loss_batch, calc_loss_loader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = dict(RUN_CONFIGS[args.run_level])
    if args.n_heads is not None:
        config["n_heads"] = args.n_heads
    if config["emb_dim"] % config["n_heads"] != 0:
        raise ValueError(
            f"emb_dim must be divisible by n_heads: emb_dim={config['emb_dim']}, n_heads={config['n_heads']}"
        )

    epochs = args.epochs if args.epochs is not None else config["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    train_chars = args.train_chars if args.train_chars is not None else config["train_chars"]
    val_chars = args.val_chars if args.val_chars is not None else config["val_chars"]
    progress = not args.no_progress
    progress_every = max(1, args.progress_every)
    progress_width = max(10, args.progress_width)
    checkpoint_every = max(0, args.checkpoint_every)
    early_stopping_patience = max(0, args.early_stopping_patience)
    early_stopping_enabled = early_stopping_patience > 0

    tokenizer, resolved_tokenizer_path = load_tokenizer(repo_root, args.tokenizer_path, config["vocab_size"])
    train_text_path = resolve_path(repo_root, args.train_text_path)
    val_text_path = resolve_path(repo_root, args.val_text_path)
    train_text = train_text_path.read_text(encoding="utf-8")[:train_chars]
    val_text = val_text_path.read_text(encoding="utf-8")[:val_chars] if val_text_path.exists() else ""
    print(f"text chars: train={len(train_text)}, val={len(val_text)}")

    train_token_ids = tokenizer.encode(train_text)
    val_token_ids = tokenizer.encode(val_text) if val_text else []
    print(f"tokens: train={len(train_token_ids)}, val={len(val_token_ids)}")

    train_loader = create_dataloader(
        train_token_ids,
        context_length=config["context_length"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if len(val_token_ids) > config["context_length"] + 1:
        val_loader = create_dataloader(
            val_token_ids,
            context_length=config["context_length"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    print(f"batches: train={len(train_loader)}, val={len(val_loader) if val_loader is not None else 0}")

    device = resolve_device(torch, args.device, args.require_cuda)
    gpt_config = make_gpt_config(config)
    model = GPTModel(gpt_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resumed_checkpoint, start_epoch, global_step, history = load_resume_checkpoint(
        model,
        optimizer,
        args.resume,
        repo_root,
        device,
    )

    run_dir = make_run_dir(
        repo_root,
        args.checkpoint_dir,
        args.run_name,
        args.run_level,
        config,
        batch_size,
        epochs,
        args.lr,
        train_chars,
        val_chars,
        args.seed,
    )
    metrics_path = resolve_path(repo_root, args.metrics_path)
    summary_path = resolve_path(repo_root, args.summary_path)
    plot_path = resolve_path(repo_root, args.plot_path)
    sample_path = resolve_path(repo_root, args.sample_path)
    for path in (metrics_path, summary_path, plot_path, sample_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append_metrics:
        for path in (metrics_path, sample_path):
            if path.exists():
                path.unlink()

    run_config_path = run_dir / "run_config.json"
    run_config = {
        "run_level": args.run_level,
        "config": gpt_config,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_chars": train_chars,
        "val_chars": val_chars,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "eval_freq": args.eval_freq,
        "eval_iter": args.eval_iter,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "patience": early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "metric": "val_loss",
        },
        "seed": args.seed,
        "tokenizer_path": str(resolved_tokenizer_path),
        "resume": str(resumed_checkpoint) if resumed_checkpoint else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checkpoint run dir: {run_dir}")
    print(f"run config: {run_config_path}")

    best_val_loss = float("inf")
    best_early_stopping_val_loss: float | None = None
    epochs_without_improvement = 0
    epochs_completed = start_epoch - 1
    early_stopped = False
    early_stopping_reason: str | None = None
    best_checkpoint_path = run_dir / "best.pt"
    final_checkpoint_path = run_dir / "final.pt"

    for epoch in range(start_epoch, epochs + 1):
        epochs_completed = epoch
        model.train()
        total_loss = 0.0
        batch_count = 0
        total_batches = len(train_loader)
        line_len = [0]

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            global_step += 1

            if batch_idx % progress_every == 0 or batch_idx == total_batches:
                avg_loss = total_loss / batch_count if batch_count else float("nan")
                print_progress(
                    progress,
                    (
                        f"epoch {epoch:03d}/{epochs:03d} "
                        f"train {format_progress_bar(batch_idx, total_batches, progress_width)} "
                        f"batch {batch_idx:04d}/{total_batches:04d} "
                        f"step={global_step} loss={avg_loss:.4f}"
                    ),
                    last_len=line_len,
                )

            if args.eval_freq and global_step % args.eval_freq == 0:
                train_eval = calc_loss_loader(train_loader, model, device, num_batches=args.eval_iter)
                val_eval = calc_loss_loader(val_loader, model, device, num_batches=args.eval_iter) if val_loader is not None else None
                row = {
                    "stage": "pretrain",
                    "event": "eval",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_eval,
                    "val_loss": val_eval,
                    "eval_iter": args.eval_iter,
                }
                history.append(row)
                append_jsonl_record(metrics_path, row)
                print_progress(
                    progress,
                    f"step {global_step}: train loss={train_eval:.4f}, val loss={val_eval:.4f}" if val_eval is not None else f"step {global_step}: train loss={train_eval:.4f}",
                    end=True,
                    last_len=line_len,
                )
        epoch_loss = total_loss / batch_count if batch_count else float("nan")
        epoch_val_loss = calc_loss_loader(val_loader, model, device, num_batches=args.eval_iter) if val_loader is not None else None
        print_progress(
            progress,
            (
                f"epoch {epoch:03d}/{epochs:03d} done "
                f"train loss={epoch_loss:.4f}, val loss={epoch_val_loss:.4f}"
                if epoch_val_loss is not None
                else f"epoch {epoch:03d}/{epochs:03d} done train loss={epoch_loss:.4f}"
            ),
            end=True,
            last_len=line_len,
        )
        epoch_row = {
            "stage": "pretrain",
            "event": "epoch",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss,
            "val_loss": epoch_val_loss,
            "eval_iter": args.eval_iter,
        }
        history.append(epoch_row)
        append_jsonl_record(metrics_path, epoch_row)

        early_stopping_improved = False
        if epoch_val_loss is not None:
            early_stopping_improved = (
                best_early_stopping_val_loss is None
                or epoch_val_loss < best_early_stopping_val_loss - args.early_stopping_min_delta
            )

        if epoch_val_loss is not None and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_pretrain_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                history=history,
                config=gpt_config,
            )
            print(f"best checkpoint saved: {best_checkpoint_path}")

        if early_stopping_enabled and epoch_val_loss is not None:
            if early_stopping_improved:
                best_early_stopping_val_loss = epoch_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(
                    f"early stopping patience: {epochs_without_improvement}/{early_stopping_patience} "
                    f"(val_loss best={best_early_stopping_val_loss:.4f}, current={epoch_val_loss:.4f})"
                )

        sample_text = ""
        if args.sample_context:
            sample_text = generate_sample(
                model,
                tokenizer,
                device,
                gpt_config,
                args.sample_context,
                args.sample_tokens,
                args.temperature,
                args.top_k,
            )
            sample_row = {
                "stage": "pretrain",
                "event": "sample",
                "epoch": epoch,
                "global_step": global_step,
                "start_context": args.sample_context,
                "generated_text": sample_text,
            }
            append_jsonl_record(sample_path, sample_row)
            print(sample_text)

        if checkpoint_every and epoch % checkpoint_every == 0:
            checkpoint_path = run_dir / f"epoch_{epoch:03d}.pt"
            save_pretrain_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                history=history,
                config=gpt_config,
            )
            print(f"epoch checkpoint saved: {checkpoint_path}")

        if early_stopping_enabled and epoch_val_loss is not None and epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            early_stopping_reason = (
                f"val_loss did not improve by at least "
                f"{args.early_stopping_min_delta:g} for {early_stopping_patience} epoch(s)"
            )
            print(f"early stopping triggered at epoch {epoch}: {early_stopping_reason}")
            break

    final_sample = generate_sample(
        model,
        tokenizer,
        device,
        gpt_config,
        args.sample_context,
        args.sample_tokens,
        args.temperature,
        args.top_k,
    ) if args.sample_context else ""
    final_result = {
        "run_level": args.run_level,
        "device": str(device),
        "config": gpt_config,
        "tokenizer_path": str(resolved_tokenizer_path),
        "train_text_path": str(train_text_path),
        "val_text_path": str(val_text_path),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": len(train_token_ids),
        "val_tokens": len(val_token_ids),
        "epochs": epochs,
        "epochs_completed": epochs_completed,
        "global_step": global_step,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "patience": early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "metric": "val_loss",
            "stopped": early_stopped,
            "reason": early_stopping_reason,
        },
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "generated_text": final_sample,
        "metrics_path": str(metrics_path),
        "sample_path": str(sample_path),
        "plot_path": str(plot_path),
        "run_config_path": str(run_config_path),
    }
    save_pretrain_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        epoch=epochs_completed,
        global_step=global_step,
        history=history,
        config=gpt_config,
        final_result=final_result,
    )
    plot_history(history, plot_path)
    summary_path.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"metrics: {metrics_path}")
    print(f"samples: {sample_path}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")
    print(f"best checkpoint: {best_checkpoint_path}")
    print(f"final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    run()
