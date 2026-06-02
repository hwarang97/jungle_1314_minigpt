# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | 작성 필요 |
| 팀명 | 작성 필요 |
| 팀원 | 작성 필요 |

---

## 1. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 담당자 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer 구현 완료 | `src/bpe.py` | 작성 필요 |
| 2 | GPTDataset, create_dataloader, InputEmbedding 구현 완료 | `src/dataset.py`, `src/embeddings.py` | 작성 필요 |
| 3 | MultiHeadAttention, causal mask 구현 완료 | `src/attention.py` | 작성 필요 |
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple 구현 완료 | `src/model.py` | 작성 필요 |
| 5 | loss 계산, checkpoint, generate, train_model 구현 완료 | `src/train.py` | 작성 필요 |
| 6 | NSMC 감성 분류 Dataset과 classifier 구현 완료 | `src/finetune.py` | 작성 필요 |

---

## 2. 테스트 통과 현황

| 실행 명령 | 결과 | 비고 |
| --- | --- | --- |
| `pytest tests/test_bpe.py -v` | 통과 | 6 passed |
| `pytest tests/test_dataset.py -v` | 통과 | 4 passed |
| `pytest tests/test_attention.py -v` | 통과 | 2 passed |
| `pytest tests/test_model.py -v` | 통과 | 7 passed |
| `pytest tests/test_train.py -v` | 통과 | 5 passed, matplotlib 비대화형 backend warning 1건 |
| `pytest tests/test_finetune.py -v` | 통과 | 4 passed |
| `pytest tests/ -v` | 통과 | 28 passed, matplotlib 비대화형 backend warning 1건 |

테스트는 원본 프로젝트의 `.venv` Python 3.11.15 환경에서 실행했습니다.
Codex 샌드박스 내부에서는 Windows 임시 폴더 권한 문제로 `tempfile.TemporaryDirectory()`를 쓰는 테스트가 실패했으나, 샌드박스 밖에서 동일 명령을 실행했을 때 전체 테스트가 통과했습니다.

| 실패한 테스트 | 에러 요약 | 해결 시도 |
| --- | --- | --- |
| 없음 | 없음 | 전체 테스트 통과 |

---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | 실제 NSMC 다운로드/학습은 미실행. Colab에서 Smoke -> Light -> Basic 순서로 확장 권장 |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | 테스트 기준 300, 제출 학습 시 Basic 예시 3000 권장 |
| 학습 corpus 크기 | 테스트는 소량 문자열, 제출 학습 시 `corpus[:1_500_000]` 권장 |
| 어휘 학습 시간 | 실제 전체 데이터 학습 미실행 |
| vocabulary 저장 경로 | 제출 학습 시 예: `data/nsmc_bpe_vocab_3000.json` |
| 인코딩/디코딩 복원 예시 | 테스트에서 한국어/영어 혼합 문장 encode -> decode 복원 확인 |

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> N x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | 실제 학습 설정 작성 필요 |
| context_length | 실제 학습 설정 작성 필요 |
| emb_dim | 실제 학습 설정 작성 필요 |
| n_heads | 실제 학습 설정 작성 필요 |
| n_layers | 실제 학습 설정 작성 필요 |
| drop_rate | 실제 학습 설정 작성 필요 |
| qkv_bias | 실제 학습 설정 작성 필요 |
| 총 파라미터 수 | 실제 학습 설정 기준 계산 필요 |

---

## 6. 사전 학습

### 6.1 하이퍼파라미터

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 모델 | vocab_size |  |
| 모델 | context_length |  |
| 모델 | emb_dim |  |
| 모델 | n_heads |  |
| 모델 | n_layers |  |
| 학습 | batch_size |  |
| 학습 | num_epochs |  |
| 학습 | eval_freq, eval_iter |  |
| 최적화 | lr, weight_decay |  |

### 6.2 결과

| 항목 | 내용 |
| --- | --- |
| train loss | epoch별 표 또는 요약 |
| validation loss | epoch별 표 또는 요약 |
| 손실 그래프 | 그래프 또는 파일 경로 |
| 생성 샘플 | 같은 시작 문맥으로 epoch별 비교 |
| checkpoint 경로 | (예: `checkpoints/ckpt_epoch_5.pt`) |

---

## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | (예: 128) |
| batch_size | (예: 16) |
| backbone learning rate |  |
| classifier learning rate |  |
| validation loss / accuracy | 실제 미세 조정 학습 후 작성 필요 |
| test loss / accuracy | 실제 미세 조정 학습 후 작성 필요 |
| 오류 예시 | 틀린 리뷰 예시와 추정 원인 |

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | 테스트 환경: Python 3.11.15 |
| PyTorch | 설치 환경 기준 |
| 실행 환경 | 로컬 단위 테스트 완료, 실제 학습은 Colab GPU 권장 |
| GPU/CPU 정보 | 실제 학습 환경 작성 필요 |
| 총 학습 소요 시간 | 실제 학습 후 작성 필요 |

---

## 9. 고찰

- 어려웠던 점
- 한국어 byte-level BPE 구현에서 조심한 점
- loss가 줄어든 이유 또는 줄어들지 않은 이유
- 과적합·과소적합 여부
- 하이퍼파라미터 변경 시도와 결과
- 다음에 개선하고 싶은 점
