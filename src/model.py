# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        # 흐름(의사코드):
        # 1. 정규화 뒤 곱할 gamma를 1로 초기화합니다.
        # 2. 정규화 뒤 더할 beta를 0으로 초기화합니다.
        # 3. 0으로 나누는 상황을 막기 위한 eps를 저장합니다.
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다.

        흐름(의사코드):
        1. 마지막 차원의 평균을 구합니다.
        2. 마지막 차원의 분산을 구합니다.
        3. (x - mean) / sqrt(var + eps)로 정규화합니다.
        4. gamma를 곱하고 beta를 더합니다.
        """
        # ('책내용') 4장: LayerNorm은 각 token 벡터의 마지막 차원을 정규화해 학습을 안정화한다.
        # batch나 sequence 차원은 유지하고, 각 token의 embedding 차원만 정규화합니다.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        tanh 근사식으로 GELU를 계산합니다.

        흐름(의사코드):
        1. x와 x^3을 이용해 GELU 근사식 내부 값을 만듭니다.
        2. tanh로 부드러운 gate 값을 만듭니다.
        3. x에 gate를 곱해 활성화 값을 반환합니다.
        """
        # ('책내용') 4장: GPT의 FFN은 ReLU보다 부드러운 GELU 활성화를 사용한다.
        return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # 흐름(의사코드):
        # 1. d_model을 mult*d_model로 확장합니다.
        # 2. GELU로 비선형성을 넣습니다.
        # 3. 다시 d_model로 줄입니다.
        # 4. dropout을 적용합니다.
        # ('책내용') 4장: FFN은 차원을 넓혔다가 다시 d_model로 줄여 각 위치의 표현력을 키운다.
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FeedForward 네트워크를 통과시킵니다. 흐름: Linear -> GELU -> Linear -> Dropout."""
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
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
        # 1. attention 앞 LayerNorm을 준비합니다.
        # 2. causal multi-head attention을 준비합니다.
        # 3. FFN 앞 LayerNorm을 준비합니다.
        # 4. FeedForward network를 준비합니다.
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """
        attention과 ffn을 residual connection으로 연결합니다.

        흐름(의사코드):
        1. x를 norm1에 통과시킨 뒤 attention을 계산합니다.
        2. attention 결과를 원래 x에 더합니다.
        3. 다시 norm2와 FeedForward를 통과시킵니다.
        4. FeedForward 결과를 원래 흐름에 더합니다.
        """
        # ('책내용') 4장: residual connection은 attention/FFN이 원래 표현 위에 필요한 변화만 더하게 한다.
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        # 흐름(의사코드):
        # 1. config를 저장합니다.
        # 2. token/position embedding 모듈을 만듭니다.
        # 3. TransformerBlock을 n_layers개 쌓습니다.
        # 4. final LayerNorm과 lm_head를 만듭니다.
        self.config = config
        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config.get("drop_rate", 0.1),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config["emb_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config.get("drop_rate", 0.1),
                    qkv_bias=config.get("qkv_bias", False),
                )
                for _ in range(config["n_layers"])
            ]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        # lm_head는 각 위치의 hidden state를 vocabulary 전체에 대한 다음 token 점수로 바꿉니다.
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)

        흐름(의사코드):
        1. input token id를 GPT hidden state로 변환합니다.
        2. lm_head로 vocab logits를 만듭니다.
        3. targets가 없으면 logits만 반환합니다.
        4. targets가 있으면 cross entropy loss도 계산해 반환합니다.
        """
        hidden = self.get_hidden_states(idx)
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(hidden)

        if targets is None:
            return logits

        # ('책내용') 5장: 모든 위치의 다음 token 예측을 한 번에 cross entropy로 계산한다.
        # cross_entropy는 class 차원이 두 번째인 (N, C)를 기대하므로 B와 T를 하나로 펼칩니다.
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, logits

    def get_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        # 흐름(의사코드):
        # 1. token id를 embedding으로 바꿉니다.
        # 2. TransformerBlock을 순서대로 통과합니다.
        # 3. 마지막 LayerNorm을 적용한 hidden state를 반환합니다.
        # fine-tuning에서도 GPT backbone의 hidden state를 재사용할 수 있도록 별도 메서드로 분리했습니다.
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x, causal_mask=True)
        return self.final_norm(x)


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다.

    흐름(의사코드):
    1. 모델을 eval mode로 바꿉니다.
    2. 최근 context_size token만 모델에 넣습니다.
    3. 마지막 위치 logits에서 argmax token을 고릅니다.
    4. 고른 token을 idx 뒤에 붙입니다.
    5. max_new_tokens만큼 반복합니다.
    6. 원래 train mode였으면 다시 train mode로 돌립니다.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # context_size보다 긴 입력은 GPT가 볼 수 있는 최근 token만 남깁니다.
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            # 다음 token 하나를 고를 때는 마지막 위치의 vocabulary logits만 사용합니다.
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
    if was_training:
        model.train()
    return idx
