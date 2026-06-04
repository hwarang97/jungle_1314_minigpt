# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path
import csv
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .metrics import append_jsonl_record
    from .model import GPTModel
except ImportError:
    from metrics import append_jsonl_record
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]

    흐름(의사코드):
    1. train TSV를 읽어 빈 리뷰를 제거합니다.
    2. seed 기준으로 train rows를 섞습니다.
    3. val_ratio만큼 validation을 분리합니다.
    4. test TSV가 있으면 test rows를 읽습니다.
    5. output_dir가 있으면 JSONL 파일로 저장합니다.
    6. train/val/test list를 반환합니다.
    """
    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        # 흐름(의사코드):
        # 1. TSV를 행 단위로 읽습니다.
        # 2. document를 text로, label을 int로 바꿉니다.
        # 3. 빈 text는 버리고 {"text", "label"} dict를 모읍니다.
        rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                if not text:
                    # NSMC에는 document가 비어 있는 행이 있을 수 있어 학습 데이터에서 제외합니다.
                    continue
                rows.append({"text": text, "label": int(row["label"])})
        return rows

    train_rows = read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    # seed를 고정한 shuffle은 train/val split이 매 실행마다 같게 나오도록 해줍니다.
    rng.shuffle(train_rows)

    val_size = int(round(len(train_rows) * val_ratio))
    if val_ratio > 0 and len(train_rows) > 1:
        # 데이터가 너무 작아도 validation이 0개 또는 train이 0개가 되지 않게 보정합니다.
        val_size = min(max(1, val_size), len(train_rows) - 1)
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Colab/로컬에서 같은 전처리를 반복하지 않도록 JSONL 파일로 저장할 수 있게 합니다.
        for name, data in {
            "nsmc_sentiment_train.jsonl": train_data,
            "nsmc_sentiment_val.jsonl": val_data,
            "nsmc_sentiment_test.jsonl": test_data,
        }.items():
            with (output_path / name).open("w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

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
        # 흐름(의사코드): 감성 분류 sample 개수를 반환합니다.
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다.

        흐름(의사코드):
        1. idx번째 리뷰와 label을 가져옵니다.
        2. tokenizer.encode로 token id를 만듭니다.
        3. max_length보다 길면 자릅니다.
        4. 짧으면 pad_id로 채웁니다.
        5. input_ids tensor와 label을 반환합니다.
        """
        item = self.data[idx]
        # ('책내용') 6장: 분류 미세조정도 먼저 문장을 token id 시퀀스로 바꾼 뒤 고정 길이 batch로 만든다.
        # BOS/EOS를 붙이면 문장의 시작과 끝 정보를 classifier가 활용할 수 있습니다.
        ids = self.tokenizer.encode(item["text"], add_bos_eos=True)[: self.max_length]
        # DataLoader가 batch로 묶으려면 모든 sample 길이가 같아야 하므로 pad_id로 채웁니다.
        ids = ids + [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long), int(item["label"])


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
        pooling: str = "last",
        classifier_head: str = "linear",
        classifier_hidden_dim: int | None = None,
    ):
        super().__init__()
        # 흐름(의사코드):
        # 1. 사전학습 GPT backbone을 저장합니다.
        # 2. 분류 class 개수를 저장합니다.
        # 3. dropout과 classifier Linear layer를 만듭니다.
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = 0
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        if classifier_head not in {"linear", "mlp"}:
            raise ValueError("classifier_head must be 'linear' or 'mlp'")
        if classifier_hidden_dim is not None and classifier_hidden_dim <= 0:
            raise ValueError("classifier_hidden_dim must be positive")
        self.pooling = pooling
        self.classifier_head = classifier_head
        self.dropout = nn.Dropout(drop_rate)
        # ('책내용') 6장: LM head는 vocab 점수용이고, 감성 분류는 num_labels 점수용 head를 따로 붙인다.
        emb_dim = gpt_model.config["emb_dim"]
        if classifier_head == "mlp":
            hidden_dim = classifier_hidden_dim or emb_dim
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.

        흐름(의사코드):
        1. GPT backbone으로 모든 위치의 hidden state를 구합니다.
        2. pad가 아닌 마지막 token 위치를 찾습니다.
        3. 그 위치의 hidden state를 문장 대표 벡터로 뽑습니다.
        4. classifier로 class logits를 만듭니다.
        5. labels가 있으면 cross entropy loss도 계산합니다.
        """
        hidden = self.gpt.get_hidden_states(input_ids)
        # pad가 아닌 token만 True가 됩니다. 문장 대표 벡터를 고를 때 padding을 제외하기 위한 mask입니다.
        non_pad = input_ids.ne(self.pad_id)
        if self.pooling == "mean":
            mask = non_pad.unsqueeze(-1)
            token_counts = non_pad.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / token_counts
        else:
            # 각 sample별 실제 token 개수를 세고, 마지막 유효 token의 index로 바꿉니다.
            last_token_idx = non_pad.sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            # GPT의 모든 위치 hidden state 중 문장 끝 위치를 문장 대표 벡터로 사용합니다.
            pooled = hidden[batch_idx, last_token_idx]
        if self.classifier_head == "mlp":
            logits = self.classifier(pooled)
        else:
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
    epoch: int | None = None,
    metrics_path: str | Path | None = "logs/sentiment_metrics.jsonl",
) -> tuple[float, float]:
    """
    감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다.

    흐름(의사코드):
    1. 모델을 train mode로 바꿉니다.
    2. batch를 device로 옮깁니다.
    3. loss를 계산하고 backward/step을 수행합니다.
    4. batch loss와 맞힌 개수를 누적합니다.
    5. 평균 loss와 accuracy를 반환합니다.
    6. metrics_path가 있으면 train loss/accuracy를 JSONL로 저장합니다.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        # loss는 평균값이라 batch 크기를 곱해 전체 sample 기준 평균을 나중에 계산합니다.
        total_loss += loss.item() * labels.size(0)
        # logits에서 점수가 가장 큰 class를 예측값으로 보고 label과 비교합니다.
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total else float("nan")
    accuracy = correct / total if total else 0.0
    append_jsonl_record(
        metrics_path,
        {
            "stage": "sentiment",
            "event": "train_epoch",
            "split": "train",
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_examples": total,
        },
    )
    return avg_loss, accuracy


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
    split: str = "eval",
    epoch: int | None = None,
    metrics_path: str | Path | None = "logs/sentiment_metrics.jsonl",
) -> tuple[float, float]:
    """
    감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다.

    흐름(의사코드):
    1. 모델을 eval mode로 바꿉니다.
    2. no_grad로 batch를 순회합니다.
    3. loss와 logits를 계산합니다.
    4. 평균 loss와 accuracy를 누적해 반환합니다.
    5. metrics_path가 있으면 평가 loss/accuracy를 JSONL로 저장합니다.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels=labels)
            # 평가에서도 train과 같은 방식으로 평균 loss와 accuracy를 누적합니다.
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total else float("nan")
    accuracy = correct / total if total else 0.0
    append_jsonl_record(
        metrics_path,
        {
            "stage": "sentiment",
            "event": "evaluate",
            "split": split,
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_examples": total,
        },
    )
    return avg_loss, accuracy
