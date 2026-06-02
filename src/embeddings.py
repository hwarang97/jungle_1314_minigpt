# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        # 흐름(의사코드):
        # 1. token id -> embedding vector layer를 만듭니다.
        # 2. position id -> position vector layer를 만듭니다.
        # 3. 두 embedding을 더한 뒤 적용할 dropout을 준비합니다.
        self.emb_dim = emb_dim
        self.context_length = context_length
        # ('책내용') 2장: token id 자체가 의미 벡터가 아니므로 학습 가능한 token embedding으로 바꾼다.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        # ('책내용') 2장: GPT는 순서를 알아야 하므로 context window 안의 위치 embedding을 더한다.
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)

        흐름(의사코드):
        1. 입력 sequence length를 확인합니다.
        2. position id 0..seq_len-1을 만듭니다.
        3. token embedding과 position embedding을 각각 구합니다.
        4. 두 embedding을 더하고 dropout을 적용합니다.
        """
        seq_len = x.shape[1]
        if seq_len > self.context_length:
            raise ValueError("seq_len cannot exceed context_length")

        # positions는 batch마다 동일하므로 (1, seq_len)으로 만들면 batch 차원에 broadcast됩니다.
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # token embedding은 "무슨 token인지", position embedding은 "몇 번째 token인지"를 담당합니다.
        x = self.token_embedding(x) + self.position_embedding(positions)
        return self.dropout(x)
