# -*- coding: utf-8 -*-
"""Analyze per-sample NSMC sentiment predictions from a fine-tuned checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
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
    },
    "BASIC": {
        "vocab_size": 3000,
        "context_length": 128,
        "emb_dim": 192,
        "n_heads": 4,
        "n_layers": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save per-sample sentiment predictions, errors, and confusion matrices.",
    )
    parser.add_argument("--run-level", choices=sorted(RUN_CONFIGS), default="BASIC")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned sentiment checkpoint path.")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/sentiment_predictions"))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def add_src_to_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_tokenizer(repo_root: Path, requested_vocab_size: int, tokenizer_path: Path | None):
    from bpe import BPETokenizer

    if tokenizer_path is not None:
        candidates = [resolve_path(repo_root, tokenizer_path)]
    else:
        candidates = [
            repo_root / "data" / f"vocab_bpe_{requested_vocab_size}.json",
            repo_root / "data" / "tokenizer.json",
        ]

    for candidate in candidates:
        if candidate.exists():
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


def load_checkpoint(path: Path, device):
    import torch

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    print(f"checkpoint: {path}")
    return checkpoint


def normalize_row(row: dict[str, Any]) -> tuple[str, int]:
    text = row.get("text", row.get("document", ""))
    label = int(row["label"])
    return text, label


def encode_for_training(tokenizer, text: str, context_length: int) -> dict[str, Any]:
    ids = tokenizer.encode(text, add_bos_eos=True)
    token_length = len(ids)
    truncated = token_length > context_length
    input_ids = ids[:context_length]
    eos_id = tokenizer.get_eos_id()
    eos_kept = eos_id in input_ids
    pad_id = tokenizer.get_pad_id()
    input_ids = input_ids + [pad_id] * (context_length - len(input_ids))
    return {
        "input_ids": input_ids,
        "token_length": token_length,
        "truncated": truncated,
        "eos_kept": eos_kept,
    }


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def summarize_records(records: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row["correct"])
    labels = [0, 1]
    confusion = {
        "true_negative": sum(1 for row in records if row["label"] == 0 and row["prediction"] == 0),
        "false_positive": sum(1 for row in records if row["label"] == 0 and row["prediction"] == 1),
        "false_negative": sum(1 for row in records if row["label"] == 1 and row["prediction"] == 0),
        "true_positive": sum(1 for row in records if row["label"] == 1 and row["prediction"] == 1),
    }

    by_label: dict[str, Any] = {}
    for label in labels:
        label_rows = [row for row in records if row["label"] == label]
        by_label[str(label)] = {
            "count": len(label_rows),
            "accuracy": safe_div(sum(1 for row in label_rows if row["correct"]), len(label_rows)),
        }

    truncated_rows = [row for row in records if row["truncated"]]
    not_truncated_rows = [row for row in records if not row["truncated"]]
    errors = [row for row in records if not row["correct"]]
    correct_rows = [row for row in records if row["correct"]]

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": row["index"],
            "text": row["text"],
            "label": row["label"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "prob_negative": row["prob_negative"],
            "prob_positive": row["prob_positive"],
            "token_length": row["token_length"],
            "truncated": row["truncated"],
            "eos_kept": row["eos_kept"],
        }

    high_confidence_errors = sorted(errors, key=lambda row: row["confidence"], reverse=True)[:top_n]
    low_confidence_errors = sorted(errors, key=lambda row: row["confidence"])[:top_n]
    longest_errors = sorted(errors, key=lambda row: row["token_length"], reverse=True)[:top_n]
    high_confidence_correct = sorted(correct_rows, key=lambda row: row["confidence"], reverse=True)[:top_n]

    return {
        "count": total,
        "accuracy": safe_div(correct, total),
        "confusion_matrix": confusion,
        "by_label": by_label,
        "truncation": {
            "truncated_count": len(truncated_rows),
            "truncated_rate": safe_div(len(truncated_rows), total),
            "truncated_accuracy": safe_div(sum(1 for row in truncated_rows if row["correct"]), len(truncated_rows)),
            "not_truncated_count": len(not_truncated_rows),
            "not_truncated_accuracy": safe_div(
                sum(1 for row in not_truncated_rows if row["correct"]),
                len(not_truncated_rows),
            ),
        },
        "top_examples": {
            "high_confidence_errors": [compact(row) for row in high_confidence_errors],
            "low_confidence_errors": [compact(row) for row in low_confidence_errors],
            "longest_errors": [compact(row) for row in longest_errors],
            "high_confidence_correct": [compact(row) for row in high_confidence_correct],
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Sentiment Prediction Analysis", ""]
    for split, split_summary in summary["splits"].items():
        trunc = split_summary["truncation"]
        cm = split_summary["confusion_matrix"]
        lines.extend(
            [
                f"## {split}",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| count | {split_summary['count']} |",
                f"| accuracy | {split_summary['accuracy']:.4f} |",
                f"| truncated rate | {trunc['truncated_rate']:.4f} |",
                f"| truncated accuracy | {trunc['truncated_accuracy']:.4f} |",
                f"| not truncated accuracy | {trunc['not_truncated_accuracy']:.4f} |",
                "",
                "| confusion | count |",
                "|---|---:|",
                f"| true_negative | {cm['true_negative']} |",
                f"| false_positive | {cm['false_positive']} |",
                f"| false_negative | {cm['false_negative']} |",
                f"| true_positive | {cm['true_positive']} |",
                "",
                "### High-confidence errors",
                "",
            ]
        )
        for row in split_summary["top_examples"]["high_confidence_errors"][:10]:
            text = row["text"].replace("\n", " ")
            lines.append(
                f"- conf={row['confidence']:.4f}, label={row['label']}, pred={row['prediction']}, "
                f"tokens={row['token_length']}, truncated={row['truncated']}: {text}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_split(
    *,
    split: str,
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    context_length: int,
    batch_size: int,
    device,
    output_dir: Path,
    top_n: int,
    progress_every: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    encoded_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text, label = normalize_row(row)
        encoded = encode_for_training(tokenizer, text, context_length)
        encoded_rows.append(
            {
                "index": index,
                "split": split,
                "text": text,
                "label": label,
                **encoded,
            }
        )

    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (start, batch) in enumerate(batched(encoded_rows, batch_size), start=1):
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=device)
            logits = model(input_ids)
            losses = F.cross_entropy(logits, labels, reduction="none")
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidences = probs.max(dim=-1).values

            for offset, row in enumerate(batch):
                prob_negative = float(probs[offset, 0].item())
                prob_positive = float(probs[offset, 1].item())
                prediction = int(predictions[offset].item())
                label = row["label"]
                records.append(
                    {
                        "index": row["index"],
                        "split": row["split"],
                        "text": row["text"],
                        "label": label,
                        "prediction": prediction,
                        "correct": prediction == label,
                        "loss": float(losses[offset].item()),
                        "prob_negative": prob_negative,
                        "prob_positive": prob_positive,
                        "confidence": float(confidences[offset].item()),
                        "token_length": row["token_length"],
                        "truncated": row["truncated"],
                        "eos_kept": row["eos_kept"],
                    }
                )

            if progress_every and (batch_idx % progress_every == 0 or start + len(batch) >= len(encoded_rows)):
                print(f"{split}: batch {batch_idx} / {(len(encoded_rows) + batch_size - 1) // batch_size}")

    predictions_path = output_dir / f"{split}_predictions.jsonl"
    write_jsonl(predictions_path, records)

    split_summary = summarize_records(records, top_n=top_n)
    split_summary["predictions_path"] = str(predictions_path)
    split_summary["avg_loss"] = safe_div(sum(row["loss"] for row in records), len(records))

    summary_path = output_dir / f"{split}_summary.json"
    summary_path.write_text(json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{split}: accuracy={split_summary['accuracy']:.4f}, loss={split_summary['avg_loss']:.4f}")
    print(f"{split}: predictions={predictions_path}")
    print(f"{split}: summary={summary_path}")
    return split_summary


def run() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    add_src_to_path(repo_root)

    import torch

    from finetune import GPTForSequenceClassification
    from model import GPTModel

    config = dict(RUN_CONFIGS[args.run_level])
    device = resolve_device(torch, args.device, args.require_cuda)
    checkpoint_path = resolve_path(repo_root, Path(args.checkpoint))
    checkpoint = load_checkpoint(checkpoint_path, device)

    checkpoint_config = checkpoint.get("gpt_config")
    expected_vocab_size = checkpoint_config["vocab_size"] if checkpoint_config else config["vocab_size"]
    tokenizer, tokenizer_path, actual_vocab_size = load_tokenizer(repo_root, expected_vocab_size, args.tokenizer_path)

    if checkpoint_config:
        gpt_config = dict(checkpoint_config)
        if gpt_config["vocab_size"] != actual_vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size={gpt_config['vocab_size']} but tokenizer vocab_size={actual_vocab_size}"
            )
    else:
        gpt_config = make_gpt_config(config, actual_vocab_size)

    backbone = GPTModel(gpt_config)
    model = GPTForSequenceClassification(backbone, num_labels=2).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"model: loaded sentiment classifier (context_length={gpt_config['context_length']})")

    data_paths = {
        "train": repo_root / "data" / "nsmc_sentiment_train.jsonl",
        "val": repo_root / "data" / "nsmc_sentiment_val.jsonl",
        "test": repo_root / "data" / "nsmc_sentiment_test.jsonl",
    }
    limits = {
        "train": args.train_limit,
        "val": args.val_limit,
        "test": args.test_limit,
    }

    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {
        "checkpoint_path": str(checkpoint_path),
        "tokenizer_path": str(tokenizer_path),
        "gpt_config": gpt_config,
        "splits": {},
    }

    for split in args.splits:
        rows = read_jsonl(data_paths[split], limit=limits[split])
        print(f"{split}: rows={len(rows)}")
        all_summary["splits"][split] = analyze_split(
            split=split,
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            context_length=gpt_config["context_length"],
            batch_size=args.batch_size,
            device=device,
            output_dir=output_dir,
            top_n=args.top_n,
            progress_every=max(0, args.progress_every),
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_dir / "summary.md"
    write_markdown_summary(markdown_path, all_summary)
    print(f"summary: {summary_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    run()
