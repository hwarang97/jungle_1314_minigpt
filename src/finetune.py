# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_nsmc(path: str | Path) -> list[dict]:
        rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                _, text, label = parts
                if not text:
                    continue
                rows.append({"text": text, "label": int(label)})
        return rows

    train_rows = read_nsmc(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    val_size = int(len(train_rows) * val_ratio)
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = read_nsmc(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        for name, rows in splits.items():
            with (output_path / f"nsmc_sentiment_{name}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        row = self.data[idx]
        ids = self.tokenizer.encode(row["text"], add_bos_eos=True)
        ids = ids[:self.max_length]
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long), int(row["label"])


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = pad_id
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        hidden = self.gpt.forward_hidden(input_ids)
        valid_mask = input_ids != self.pad_id
        last_indices = valid_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = hidden[batch_indices, last_indices]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        loss = F.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += input_ids.size(0)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels=labels)
            total_loss += loss.item() * input_ids.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += input_ids.size(0)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def save_sentiment_history(history: list[dict], path: str | Path) -> None:
    """감성 분류 실험 history를 그래프용 JSONL 또는 CSV 파일로 저장합니다."""
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


def train_sentiment_model(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    val_loader=None,
    test_loader=None,
    metrics_path: str | Path | None = None,
) -> list[dict]:
    """감성 분류 모델을 학습하고 epoch별 loss/accuracy history를 반환합니다."""
    history = []
    model.to(device)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch_sentiment(
            model,
            train_loader,
            optimizer,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }

        if val_loader is not None:
            val_loss, val_accuracy = evaluate_sentiment(model, val_loader, device)
            row.update({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })

        if test_loader is not None:
            test_loss, test_accuracy = evaluate_sentiment(model, test_loader, device)
            row.update({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            })

        history.append(row)
        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}"
            + (f", val_loss={row['val_loss']:.4f}, val_acc={row['val_accuracy']:.4f}" if "val_loss" in row else "")
            + (f", test_loss={row['test_loss']:.4f}, test_acc={row['test_accuracy']:.4f}" if "test_loss" in row else "")
        )

        if metrics_path is not None:
            save_sentiment_history(history, metrics_path)

    return history
