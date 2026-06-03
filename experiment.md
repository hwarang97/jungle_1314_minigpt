# MiniGPT Sentiment Fine-tuning Report

## 1. 목적
- NSMC sentiment classification 성능 확인
- 사전학습 checkpoint 사용 여부 비교
- validation loss/accuracy 기준으로 과적합 여부 분석

## 2. Vocabulary / Tokenizer

### 2.1 Vocab 생성 방식
| 항목 | 값 |
|---|---:|
| tokenizer type | UTF-8 byte-level BPE |
| 구현 파일 | src/bpe.py |
| 저장 파일 | data/tokenizer.json |
| 파일 크기 | 318,919 bytes |
| vocab_size | 3,000 |
| special tokens | 4 |
| byte tokens | 256 |
| BPE merge tokens | 2,740 |
| merge rules | 2,740 |

`tokenizer.json`은 NSMC language modeling corpus를 바탕으로 학습한 vocabulary와 merge rule을 저장한 파일이다.
외부 tokenizer vocabulary를 사용하지 않고, 직접 구현한 UTF-8 byte-level BPE tokenizer를 사용했다.

### 2.2 Vocab 사용 방식
`tokenizer.json`은 텍스트를 token id로 바꾸기 위한 고정된 사전이며, 모델 가중치는 아니다.
따라서 token embedding table은 `tokenizer.json`에 들어 있지 않고, 사전학습과 미세튜닝 과정에서 모델이 직접 학습한다.

사전학습과 미세튜닝에서는 같은 tokenizer를 사용했다.
이렇게 해야 사전학습된 checkpoint의 embedding table과 미세튜닝 입력 token id가 같은 의미를 유지할 수 있다.

## 3. 현재 데이터셋
| split | 사용 개수 | 전체 개수 | 비율 |
|---|---:|---:|---:|
| train | 35,000 | 137,996 | 약 25.4% |
| validation | 6,000 | 11,999 | 약 50.0% |
| test | 5,000 | 49,997 | 약 10.0% |

## 4. 공통 설정
- run_level: BASIC
- emb_dim: 192
- n_layers: 4
- n_heads: 4
- context_length: 128
- tokenizer: data/tokenizer.json
- pretrained checkpoint: checkpoints/pretrain/.../best.pt
- optimizer: AdamW
- batch_size: 8
- weight_decay: 0.1
- seed: 42
- early stopping: val_loss, patience 2, min_delta 0.001

## 5. 사전학습

### 5.1 사전학습 설정
| 항목 | 값 |
|---|---:|
| task | NSMC language modeling |
| train data | data/nsmc_lm_train.txt |
| validation data | data/nsmc_lm_val.txt |
| tokenizer | data/tokenizer.json |
| run_level | BASIC |
| vocab_size | 3,000 |
| context_length | 128 |
| emb_dim | 192 |
| n_layers | 4 |
| n_heads | 4 |
| batch_size | 32 |
| epochs | 20 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight_decay | 0.01 |
| seed | 42 |
| device | CUDA |

### 5.2 사전학습 데이터와 결과
| 항목 | 값 |
|---|---:|
| requested train chars | 1,500,000 |
| actual train chars | 1,379,486 |
| train tokens | 3,335,336 |
| requested validation chars | 200,000 |
| actual validation chars | 120,560 |
| validation tokens | 291,753 |
| global steps | 16,300 |
| final train loss | 1.3138 |
| best validation loss | 1.2911 |
| best checkpoint | checkpoints/pretrain/BASIC_ctx128_emb192_L4_H4_bs32_lr0.0003_chars1500000_val200000_ep20_seed42_20260603_153345/best.pt |
| final checkpoint | checkpoints/pretrain/BASIC_ctx128_emb192_L4_H4_bs32_lr0.0003_chars1500000_val200000_ep20_seed42_20260603_153345/final.pt |

### 5.3 사전학습 그래프
![pretrain basic epoch loss](figures/pretrain_basic_epoch_loss.png)

### 5.4 사전학습 결과 분석
그래프의 경향을 보면 train loss와 validation loss가 함께 감소하고 있어, 뚜렷한 과적합 없이 이상적으로 학습이 진행되고 있다고 판단했다.
따라서 이 단계에서는 사전학습 성능을 더 끌어올리기보다, 사전학습된 checkpoint를 사용해 미세튜닝으로 넘어가서 sentiment classification 성능을 개선하는 것이 더 적절하다고 판단했다.

### 5.5 사전학습 결과
![LLM](figures/LLM.png)

## 6. 실험 기록
| 실험 | lr | epochs | completed | best epoch | best val loss | best val acc | test acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1e-4 | 10 | 10 | 10 | 0.5353 | 0.7900 | 0.8032 |
| low lr | 1e-5 | 10 | 10 | 8 | 0.4818 | 0.7680 | 0.7896 |
| mid lr | 5e-5 | 20 | 6 | 4 | 0.4590 | 0.7855 | 0.8012 |

## 7. Fine-tuning 결과
![sentimental](figures/sentimental%20predict%20result.png)

### 7.1 초기 fine-tuning 결과
![lr1e-4 result](figures/sentiment_pretrained_train35000_ep10.png)

### 7.2 개선사항 적용 결과

#### 7.2.1 lr=1e-5
![lr1e-5 result](figures/sentiment_pretrained_train35000_ep10_lr1e-5.png)

#### 7.2.2 lr=5e-5
![lr5e-5 result](figures/sentiment_pretrained_train35000_ep20_lr5e-5.png)

#### 7.2.2 dropout=0.2 
![dropout0.2 result](figures/dropout0.2%20result.png)

#### 7.2.3 dropout=0.3
![dropout0.3 result]

## 8. 분석
실성능은 80% 내외에서 머무르고 있다.
train accuracy는 계속 상승하지만 validation/test 성능은 크게 개선되지 않아, train 데이터에 대한 과적합이 발생하고 있는 것으로 보인다.
따라서 lr과 dropout을 조정해 결과가 개선되는지 재확인해보기로 했다.

## 9. 관찰
- train loss는 감소했는가?
- validation loss는 언제부터 증가했는가?
- validation accuracy는 개선되었는가?
- early stopping이 어느 epoch에서 걸렸는가?

## 10. 해석
- 과적합 가능성
- lr 영향
- 데이터 크기 영향
- 사전학습 checkpoint 효과

## 11. 개선 방향
- vocab_size를 3,000에서 6,000 / 8,000 / 12,000으로 늘려 평균 token 길이와 truncation 비율을 비교한다.
- context_length 128에서 약 16~18%의 리뷰가 잘리고 있으므로, context_length 256 실험을 검토한다.
- truncation 시 EOS token이 사라지지 않도록 `BOS + 앞부분 + EOS` 형태를 보장한다.
- 사전학습 validation loss가 epoch 20까지 계속 감소했으므로, vocab/context 변경 후 사전학습을 더 길게 진행한다.
- 현재 분류 head는 마지막 유효 token hidden state만 사용하므로, mean pooling 또는 EOS pooling을 비교한다.
