# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # 흐름(의사코드):
        # 1. d_model이 n_heads로 나누어지는지 확인합니다.
        # 2. head_dim을 계산합니다.
        # 3. Q/K/V projection layer를 준비합니다.
        # 4. head를 합친 뒤 쓸 output projection과 dropout을 준비합니다.
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # ('책내용') 3장: Q는 찾는 정보, K는 비교 기준, V는 실제 전달할 정보로 투영한다.
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop_rate)
        self.out_dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        multi-head attention forward를 수행합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환

        흐름(의사코드):
        1. x에서 Q, K, V를 만듭니다.
        2. Q/K/V를 여러 head로 나눕니다.
        3. QK^T / sqrt(head_dim)으로 score를 계산합니다.
        4. causal mask가 켜져 있으면 미래 위치를 -inf로 가립니다.
        5. softmax로 attention weight를 만듭니다.
        6. attention weight와 V를 곱합니다.
        7. head를 다시 합치고 output projection을 적용합니다.
        """
        batch_size, seq_len, _ = x.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            # 흐름(의사코드): (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
            # d_model 전체 차원을 n_heads개로 나누어 각 head가 head_dim만큼 담당하게 합니다.
            tensor = tensor.view(batch_size, seq_len, self.n_heads, self.head_dim)
            # attention 계산은 head별로 독립 수행하므로 head 차원을 sequence 차원 앞으로 옮깁니다.
            return tensor.transpose(1, 2)

        # 입력 x는 같은 값이지만, 서로 다른 Linear layer를 지나 Q/K/V 역할로 분리됩니다.
        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # ('책내용') 3장: QK^T를 sqrt(head_dim)으로 나눠 softmax가 너무 뾰족해지는 것을 막는다.
        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)

        if causal_mask:
            # ('책내용') 3장: GPT는 다음 token 예측 모델이라 미래 token을 보면 정답을 미리 보는 셈이다.
            # diagonal=1은 자기 자신은 볼 수 있게 두고, 그 오른쪽 미래 위치만 True로 만듭니다.
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            # softmax 전에 -inf로 바꾸면 해당 위치의 attention weight가 0이 됩니다.
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        # 각 token이 참고하기로 한 비율(attn_weights)로 V 정보를 섞습니다.
        context = attn_weights @ v
        # head별 결과를 다시 하나의 d_model 벡터로 합쳐 다음 layer가 받을 shape로 되돌립니다.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_dropout(self.out_proj(context))

        if return_attention_weights:
            return out, attn_weights
        return out
