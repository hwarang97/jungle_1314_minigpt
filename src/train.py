# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
from pathlib import Path

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
    """
    한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다.

    흐름(의사코드):
    1. input_batch와 target_batch를 device로 옮깁니다.
    2. model(input, targets=target)을 호출합니다.
    3. 모델이 계산한 loss를 반환합니다.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # ('책내용') 5장: 한 배치 학습은 입력을 device로 옮긴 뒤 loss->backward->step 순서로 진행된다.
    # GPTModel은 targets가 있으면 내부에서 logits와 cross entropy loss를 함께 계산합니다.
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """
    data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용합니다.

    흐름(의사코드):
    1. 현재 train/eval mode를 기억합니다.
    2. eval mode와 no_grad로 batch loss를 계산합니다.
    3. 지정된 num_batches만큼 또는 loader 끝까지 반복합니다.
    4. 평균 loss를 반환합니다.
    5. 원래 train mode였으면 다시 train mode로 복구합니다.
    """
    was_training = model.training
    # 평가 중에는 dropout 같은 학습 전용 동작을 끄기 위해 eval mode로 전환합니다.
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            count += 1

    if was_training:
        # 호출 전 학습 모드였던 모델은 평가 후 다시 train mode로 돌려놓습니다.
        model.train()
    return total_loss / count if count else float("nan")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """
    model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다.

    흐름(의사코드):
    1. checkpoint 폴더를 준비합니다.
    2. model.state_dict()와 optimizer.state_dict()를 묶습니다.
    3. epoch, global_step을 함께 저장합니다.
    4. torch.save로 파일에 기록합니다.
    """
    path_obj = Path(path)
    if path_obj.parent != Path("."):
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    # ('책내용') 5장: checkpoint에는 이어 학습할 수 있도록 optimizer 상태와 step도 함께 저장한다.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path_obj,
    )


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """
    torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다.

    흐름(의사코드):
    1. checkpoint를 device에 맞게 읽습니다.
    2. model state를 복원합니다.
    3. optimizer가 있으면 optimizer state도 복원합니다.
    4. epoch와 global_step을 반환합니다.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        # optimizer 상태를 복원해야 AdamW의 내부 추정값까지 이어서 사용할 수 있습니다.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    temperature와 top-k 샘플링을 지원하는 생성 함수입니다.

    흐름(의사코드):
    1. 모델을 eval mode로 바꿉니다.
    2. 최근 context_size token을 모델에 넣습니다.
    3. 마지막 위치 logits를 꺼냅니다.
    4. top_k가 있으면 후보 token을 상위 k개로 제한합니다.
    5. temperature가 0이면 argmax, 아니면 softmax sampling을 합니다.
    6. next_id를 입력 뒤에 붙입니다.
    7. eos_id가 나오면 멈출 수 있습니다.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 매 step마다 모델의 context_length를 넘지 않도록 최근 token만 입력합니다.
            idx_cond = idx[:, -context_size:]
            # 다음 token 선택에는 sequence의 마지막 위치 logits만 필요합니다.
            logits = model(idx_cond)[:, -1, :]

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, k)
                # top-k 밖의 token은 -inf로 막아 softmax 확률이 0이 되게 합니다.
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))

            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # ('책내용') 5장: temperature/top-k는 다음 token 후보 분포의 다양성을 조절한다.
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and torch.all(next_id == eos_id):
                # batch의 모든 샘플이 eos를 생성하면 더 이어 쓸 필요가 없습니다.
                break

    if was_training:
        model.train()
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
    """
    start_context를 encode하고 generate 후 decode하여 출력합니다.

    흐름(의사코드):
    1. 시작 문장을 tokenizer.encode로 token id로 바꿉니다.
    2. batch 차원을 추가합니다.
    3. generate로 새 token을 이어 붙입니다.
    4. tokenizer.decode로 사람이 읽는 문자열로 복원해 출력합니다.
    """
    model.eval()
    encoded = tokenizer.encode(start_context, add_bos_eos=False)
    # generate는 batch 입력을 기대하므로 시작 token 목록에 batch 차원을 하나 추가합니다.
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    out = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=getattr(tokenizer, "get_eos_id", lambda: None)(),
    )
    print(tokenizer.decode(out[0].tolist(), skip_special=True))


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
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """
    사전 학습 루프를 실행하고 epoch별 train loss 리스트를 반환합니다.

    흐름(의사코드):
    1. 모델을 device로 옮깁니다.
    2. epoch를 반복합니다.
    3. batch마다 zero_grad -> loss -> backward -> step을 수행합니다.
    4. eval_freq마다 train/val loss를 출력합니다.
    5. ckpt_freq마다 checkpoint를 저장합니다.
    6. epoch 평균 train loss를 기록합니다.
    7. start_context가 있으면 샘플 생성을 출력합니다.
    """
    model.to(device)
    train_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for input_batch, target_batch in train_loader:
            # PyTorch는 gradient를 누적하므로 매 batch 시작 전에 이전 gradient를 지웁니다.
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # loss가 줄어드는 방향으로 각 파라미터의 gradient를 계산합니다.
            loss.backward()
            # optimizer가 계산된 gradient를 이용해 모델 파라미터를 한 번 업데이트합니다.
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            global_step += 1

            if eval_freq and global_step % eval_freq == 0:
                # eval_iter만큼만 평가하면 큰 validation set에서도 중간 확인 시간을 줄일 수 있습니다.
                train_eval = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                if val_loader is not None:
                    val_eval = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                    print(f"step {global_step}: train loss {train_eval:.4f}, val loss {val_eval:.4f}")
                else:
                    print(f"step {global_step}: train loss {train_eval:.4f}")

            if ckpt_freq and global_step % ckpt_freq == 0:
                # 긴 Colab 학습이 끊겨도 이어갈 수 있도록 주기적으로 checkpoint를 저장합니다.
                save_checkpoint(model, optimizer, epoch=epoch, global_step=global_step, path=f"checkpoints/ckpt_step_{global_step}.pt")

        train_losses.append(total_loss / batch_count if batch_count else float("nan"))
        if start_context:
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context,
                max_new_tokens=20,
                context_size=model.config["context_length"],
            )

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """
    훈련/검증 손실 그래프를 그리는 제공 함수.

    흐름(의사코드):
    1. train loss 선을 그립니다.
    2. val loss가 있으면 함께 그립니다.
    3. 축 이름, 범례, 제목을 붙이고 화면에 표시합니다.
    """
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
