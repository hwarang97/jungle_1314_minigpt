# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        # 흐름(의사코드):
        # 1. token_ids, context_length, stride를 저장합니다.
        # 2. context_length와 stride가 올바른지 검사합니다.
        # 3. 만들 수 있는 input/target 샘플 수를 계산합니다.
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        # ('책내용') 2장: GPT는 현재 token 구간으로 다음 token을 맞히므로 target까지 context_length+1개가 필요하다.
        # len=100, context=10이면 input 10개와 그 다음 target 10개를 만들기 위해 마지막 token 1개를 더 봅니다.
        usable = len(self.token_ids) - self.context_length - 1
        # stride 간격으로 시작점을 이동할 수 있는 횟수가 전체 학습 샘플 개수입니다.
        self._length = max(0, usable // self.stride + 1)

    def __len__(self) -> int:
        """전체 샘플 개수를 반환합니다. 흐름: 미리 계산한 self._length를 돌려줍니다."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)

        흐름(의사코드):
        1. idx가 범위 안인지 확인합니다.
        2. start = idx * stride로 window 시작점을 계산합니다.
        3. input은 start부터 context_length만큼 자릅니다.
        4. target은 start+1부터 같은 길이만큼 자릅니다.
        5. 둘 다 LongTensor로 반환합니다.
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        start = idx * self.stride
        end = start + self.context_length
        # input은 현재 context window입니다.
        input_ids = self.token_ids[start:end]
        # target은 같은 길이지만 시작점을 한 칸 뒤로 밀어 "다음 token" 정답을 만듭니다.
        target_ids = self.token_ids[start + 1 : end + 1]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다.

    흐름(의사코드):
    1. token_ids로 GPTDataset을 만듭니다.
    2. batch_size, shuffle, drop_last 옵션을 DataLoader에 넘깁니다.
    3. 학습 루프가 바로 쓸 수 있는 DataLoader를 반환합니다.
    """
    dataset = GPTDataset(token_ids, context_length=context_length, stride=stride)
    # DataLoader는 학습 루프가 바로 사용할 수 있게 (input_batch, target_batch)를 batch 단위로 묶어줍니다.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
