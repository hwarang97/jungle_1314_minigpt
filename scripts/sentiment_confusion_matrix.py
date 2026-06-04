# -*- coding: utf-8 -*-
"""Evaluate a fine-tuned sentiment checkpoint with confusion matrices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


LABEL_NAMES = {
    0: "negative",
    1: "positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a sentiment checkpoint and write val/test confusion matrices.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--pooling", choices=["last", "mean"], default=None)
    parser.add_argument("--classifier-head", choices=["linear", "mlp"], default=None)
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive probability threshold.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/sentiment_confusion"))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def add_src_to_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_config(repo_root: Path, checkpoint_path: Path, config_path: Path | None) -> dict[str, Any]:
    if config_path is not None:
        return load_json(resolve_path(repo_root, config_path))

    candidate = checkpoint_path.parent / "run_config.json"
    if candidate.exists():
        return load_json(candidate)
    return {}


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


def limit_for_split(args: argparse.Namespace, run_config: dict[str, Any], split: str) -> int | None:
    explicit = {
        "train": args.train_limit,
        "val": args.val_limit,
        "test": args.test_limit,
    }[split]
    if explicit is not None:
        return explicit
    return run_config.get("limits", {}).get(split)


def load_split(repo_root: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    name = {
        "train": "nsmc_sentiment_train.jsonl",
        "val": "nsmc_sentiment_val.jsonl",
        "test": "nsmc_sentiment_test.jsonl",
    }[split]
    return read_jsonl(repo_root / "data" / name, limit)


def load_tokenizer(repo_root: Path, tokenizer_path: Path | None, run_config: dict[str, Any], vocab_size: int):
    from bpe import BPETokenizer

    if tokenizer_path is None and run_config.get("tokenizer_path"):
        tokenizer_path = Path(run_config["tokenizer_path"])
    if tokenizer_path is None:
        tokenizer_path = Path("data/tokenizer.json")

    resolved = resolve_path(repo_root, tokenizer_path)
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.load(resolved)
    actual_vocab_size = len(tokenizer.id_to_token)
    print(f"tokenizer: {resolved} (vocab_size={actual_vocab_size})")
    return tokenizer, resolved, actual_vocab_size


def infer_pooling(args: argparse.Namespace, run_config: dict[str, Any]) -> str:
    if args.pooling is not None:
        return args.pooling
    return str(run_config.get("pooling") or "last")


def infer_classifier_head(args: argparse.Namespace, run_config: dict[str, Any]) -> str:
    if args.classifier_head is not None:
        return args.classifier_head
    return str(run_config.get("classifier_head") or "linear")


def infer_classifier_hidden_dim(args: argparse.Namespace, run_config: dict[str, Any]) -> int | None:
    if args.classifier_hidden_dim is not None:
        return args.classifier_hidden_dim
    hidden_dim = run_config.get("classifier_hidden_dim")
    return int(hidden_dim) if hidden_dim is not None else None


def load_model(repo_root: Path, args: argparse.Namespace, run_config: dict[str, Any], device):
    import torch

    from finetune import GPTForSequenceClassification
    from model import GPTModel

    checkpoint_path = resolve_path(repo_root, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gpt_config = checkpoint.get("gpt_config") or run_config.get("gpt_config")
    if gpt_config is None:
        raise ValueError("Could not find gpt_config in checkpoint or run_config.json.")

    tokenizer, tokenizer_path, actual_vocab_size = load_tokenizer(
        repo_root,
        args.tokenizer_path,
        run_config,
        vocab_size=int(gpt_config["vocab_size"]),
    )
    gpt_config = dict(gpt_config)
    gpt_config["vocab_size"] = actual_vocab_size

    pooling = infer_pooling(args, run_config)
    classifier_head = infer_classifier_head(args, run_config)
    classifier_hidden_dim = infer_classifier_hidden_dim(args, run_config)
    classifier_dropout = float(run_config.get("classifier_dropout", 0.1))
    model = GPTForSequenceClassification(
        GPTModel(gpt_config),
        num_labels=2,
        drop_rate=classifier_dropout,
        pooling=pooling,
        classifier_head=classifier_head,
        classifier_hidden_dim=classifier_hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint epoch: {epoch}")
    print(f"pooling: {pooling}")
    print(f"classifier_head: {classifier_head}")
    print(f"classifier_hidden_dim: {classifier_hidden_dim}")
    return model, tokenizer, gpt_config, tokenizer_path, checkpoint


def compute_confusion(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 1 and pred == 1:
            tp += 1

    total = tn + fp + fn + tp
    actual_negative = tn + fp
    actual_positive = fn + tp
    predicted_negative = tn + fn
    predicted_positive = fp + tp

    def safe_div(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    return {
        "matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": safe_div(tn + tp, total),
        "actual_negative": actual_negative,
        "actual_positive": actual_positive,
        "predicted_negative": predicted_negative,
        "predicted_positive": predicted_positive,
        "negative_recall": safe_div(tn, actual_negative),
        "positive_recall": safe_div(tp, actual_positive),
        "negative_precision": safe_div(tn, predicted_negative),
        "positive_precision": safe_div(tp, predicted_positive),
        "false_positive_rate": safe_div(fp, actual_negative),
        "false_negative_rate": safe_div(fn, actual_positive),
    }


def evaluate_split(model, dataset, batch_size: int, num_workers: int, device, threshold: float):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    positive_probs: list[float] = []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=-1)
            pos_probs = probs[:, 1]
            preds = (pos_probs >= threshold).long()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            positive_probs.extend(pos_probs.cpu().tolist())

    return compute_confusion(y_true, y_pred), y_true, y_pred, positive_probs


def write_confusion_csv(path: Path, confusion: dict[str, Any]) -> None:
    matrix = confusion["matrix"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", "negative", "positive"])
        writer.writerow(["negative", matrix[0][0], matrix[0][1]])
        writer.writerow(["positive", matrix[1][0], matrix[1][1]])


def write_predictions_csv(
    path: Path,
    rows: list[dict[str, Any]],
    y_pred: list[int],
    positive_probs: list[float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "label_name", "prediction", "prediction_name", "positive_probability", "text"])
        for idx, (row, pred, pos_prob) in enumerate(zip(rows, y_pred, positive_probs)):
            label = int(row["label"])
            writer.writerow([
                idx,
                label,
                LABEL_NAMES[label],
                pred,
                LABEL_NAMES[pred],
                f"{pos_prob:.8f}",
                row["text"],
            ])


def write_plot(path: Path, split: str, confusion: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    matrix = confusion["matrix"]
    total = max(1, confusion["total"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"{split} confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])

    for row_idx in range(2):
        for col_idx in range(2):
            count = matrix[row_idx][col_idx]
            pct = count / total * 100
            ax.text(
                col_idx,
                row_idx,
                f"{count:,}\n{pct:.1f}%",
                ha="center",
                va="center",
                color="white" if count > total * 0.18 else "black",
                fontsize=11,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def format_summary(split: str, confusion: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## {split}",
            "",
            f"- total: {confusion['total']:,}",
            f"- accuracy: {confusion['accuracy']:.4f}",
            f"- actual negative / positive: {confusion['actual_negative']:,} / {confusion['actual_positive']:,}",
            f"- predicted negative / positive: {confusion['predicted_negative']:,} / {confusion['predicted_positive']:,}",
            f"- negative recall: {confusion['negative_recall']:.4f}",
            f"- positive recall: {confusion['positive_recall']:.4f}",
            f"- negative precision: {confusion['negative_precision']:.4f}",
            f"- positive precision: {confusion['positive_precision']:.4f}",
            "",
            "| actual \\ predicted | negative | positive |",
            "|---|---:|---:|",
            f"| negative | {confusion['tn']:,} | {confusion['fp']:,} |",
            f"| positive | {confusion['fn']:,} | {confusion['tp']:,} |",
            "",
        ]
    )


def run() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    add_src_to_path(repo_root)

    import torch

    from finetune import ReviewSentimentDataset

    checkpoint_path = resolve_path(repo_root, args.checkpoint)
    run_config = load_run_config(repo_root, checkpoint_path, args.config_path)
    device = resolve_device(torch, args.device, args.require_cuda)
    model, tokenizer, gpt_config, tokenizer_path, checkpoint = load_model(repo_root, args, run_config, device)

    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "tokenizer_path": str(tokenizer_path),
        "pooling": infer_pooling(args, run_config),
        "classifier_head": infer_classifier_head(args, run_config),
        "classifier_hidden_dim": infer_classifier_hidden_dim(args, run_config),
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "splits": {},
    }
    markdown_parts = [
        "# Sentiment Confusion Matrix",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint epoch: {checkpoint.get('epoch')}",
        f"- pooling: `{summary['pooling']}`",
        f"- classifier_head: `{summary['classifier_head']}`",
        f"- classifier_hidden_dim: `{summary['classifier_hidden_dim']}`",
        f"- threshold: {args.threshold}",
        "",
    ]

    for split in args.splits:
        limit = limit_for_split(args, run_config, split)
        rows = load_split(repo_root, split, limit)
        print(f"{split}: examples={len(rows)}")
        dataset = ReviewSentimentDataset(rows, tokenizer, max_length=int(gpt_config["context_length"]))
        confusion, _, y_pred, positive_probs = evaluate_split(
            model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            threshold=args.threshold,
        )
        summary["splits"][split] = confusion

        write_confusion_csv(output_dir / f"{split}_confusion_matrix.csv", confusion)
        write_predictions_csv(output_dir / f"{split}_predictions.csv", rows, y_pred, positive_probs)
        if not args.no_plot:
            write_plot(output_dir / f"{split}_confusion_matrix.png", split, confusion)
        markdown_parts.append(format_summary(split, confusion))

        print(
            f"{split}: acc={confusion['accuracy']:.4f}, "
            f"TN={confusion['tn']}, FP={confusion['fp']}, "
            f"FN={confusion['fn']}, TP={confusion['tp']}"
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text("\n".join(markdown_parts), encoding="utf-8")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    run()
