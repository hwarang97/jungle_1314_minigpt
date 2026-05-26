# Project Notes

## 프로젝트 이해

이 프로젝트는 PyTorch만 사용해 작은 GPT 계열 언어 모델을 직접 구현하는 학생용 템플릿이다. 목표는 ChatGPT 같은 대형 모델을 만드는 것이 아니라, LLM을 구성하는 핵심 component를 직접 구현하면서 동작 원리를 이해하는 교육용 mini GPT를 완성하는 것이다.

현재 학생용 소스에는 `TODO`와 `NotImplementedError`가 남아 있다. 처음부터 전체 테스트를 실행하면 실패하는 것이 정상이며, 각 단계의 TODO를 구현한 뒤 해당 단계의 테스트 파일을 하나씩 통과시키는 방식으로 진행한다.

참고 자료는 『밑바닥부터 만들면서 배우는 LLM』과 교재 소스 코드 `https://github.com/rickiepark/llm-from-scratch`이다. 다만 이 과제에서는 외부 tokenizer와 pretrained model을 사용하지 않고, tokenizer까지 직접 구현한다.

## 기술 제약

- Python 3.11을 기준으로 한다.
- 허용 라이브러리는 `torch`, `torch.nn`, `torch.utils.data`, `numpy`, `matplotlib`, `pytest`이다.
- Hugging Face `transformers`, `datasets`, `tokenizers`, `sentencepiece`, `spacy`, `nltk`, `lightning`, `accelerate`는 사용하지 않는다.
- 외부 pretrained model, 외부 tokenizer vocabulary, `tiktoken`도 사용하지 않는다.

## 주요 구성

| 파일 | 역할 |
| --- | --- |
| `download_data.py` | NSMC 원본 데이터를 내려받고 과제용 파일을 생성 |
| `gpt-lab.ipynb` | Colab/로컬 실행 순서 안내 노트북 |
| `src/bpe.py` | UTF-8 byte-level BPE tokenizer |
| `src/dataset.py` | GPT 사전 학습용 Dataset과 DataLoader |
| `src/embeddings.py` | token embedding + position embedding |
| `src/attention.py` | causal multi-head self-attention |
| `src/model.py` | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel |
| `src/train.py` | loss 계산, checkpoint, generation, pretraining loop |
| `src/finetune.py` | NSMC 감성 분류 Dataset과 classifier |
| `tests/` | 단계별 구현 검증 테스트 |

## 데이터

기본 데이터는 NAVER Sentiment Movie Corpus(NSMC)이다.

- 원본 저장소: `https://github.com/e9t/nsmc`
- 라이선스: CC0 1.0
- 원본 파일: `ratings_train.txt`, `ratings_test.txt`
- 컬럼: `id`, `document`, `label`

`python download_data.py`를 실행하면 다음 과제용 파일이 생성된다.

- `data/nsmc_lm_train.txt`
- `data/nsmc_lm_val.txt`
- `data/nsmc_sentiment_train.jsonl`
- `data/nsmc_sentiment_val.jsonl`
- `data/nsmc_sentiment_test.jsonl`

데이터 파일, checkpoint, token, 비밀번호는 GitHub에 commit하지 않는다.

## 구현 순서

| 순서 | 구현 대상 | 파일 | 테스트 |
| --- | --- | --- | --- |
| 1 | BPE tokenizer | `src/bpe.py` | `pytest tests/test_bpe.py -v` |
| 2 | Dataset / InputEmbedding | `src/dataset.py`, `src/embeddings.py` | `pytest tests/test_dataset.py -v` |
| 3 | MultiHeadAttention | `src/attention.py` | `pytest tests/test_attention.py -v` |
| 4 | GPT 모델 구성 요소 | `src/model.py` | `pytest tests/test_model.py -v` |
| 5 | 사전 학습 유틸리티 | `src/train.py` | `pytest tests/test_train.py -v` |
| 6 | 감성 분류 미세 조정 | `src/finetune.py` | `pytest tests/test_finetune.py -v` |
| 7 | 전체 테스트 | 전체 | `pytest tests/ -v` |

처음부터 전체 테스트만 실행하면 실패 원인을 찾기 어렵다. 현재 구현 중인 단계의 테스트부터 실행하고, 단계별 테스트가 모두 통과한 뒤 마지막에 전체 테스트를 실행한다.

## 로컬 실행

```bash
conda create -n gpt-lab python=3.11 -y
conda activate gpt-lab
pip install -r requirements.txt
pytest tests/ -v
```

## 선택 과제

필수 구현을 마친 뒤에는 학습률 warmup, cosine decay, gradient clipping, weight decay, 하이퍼파라미터 탐색, 감성 분류 성능 개선을 실험할 수 있다.
