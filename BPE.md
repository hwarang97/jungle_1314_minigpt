# BPE Tokenizer 정리

## 1. BPE의 역할

BPE(Byte Pair Encoding)는 텍스트를 모델이 처리할 수 있는 **token ID 리스트**로 바꾸는 tokenizer 알고리즘이다.

이 프로젝트의 BPE는 의미를 이해하거나 단어 벡터를 학습하지 않는다. BPE가 하는 일은 다음과 같다.

```text
문자열
-> UTF-8 byte sequence
-> byte token ID sequence
-> 자주 나오는 token pair를 merge
-> 최종 token ID sequence
```

예를 들어 `"가"`는 UTF-8에서 3개의 byte로 표현된다.

```text
"가"
-> [234, 176, 128]
```

이 프로젝트에서는 special token 4개가 ID `0~3`을 사용하므로, byte token은 `4~259`를 사용한다.

```text
byte 234 -> token id 238
byte 176 -> token id 180
byte 128 -> token id 132
```

따라서 처음에는 `"가"`가 이렇게 표현된다.

```text
"가"
-> [238, 180, 132]
```

BPE 학습을 하면 자주 붙어 나오는 token pair를 하나의 새 token으로 합친다.

```text
(238, 180) -> 260
(260, 132) -> 261
```

그러면 `"가"`를 더 짧게 표현할 수 있다.

```text
[238, 180, 132]
-> [260, 132]
-> [261]
```

즉 BPE의 핵심 역할은 **자주 나오는 byte/token 조합을 더 큰 token으로 묶어서 sequence 길이를 줄이는 것**이다.

## 2. BPE가 필요한 이유

문자열을 byte 단위로만 처리하면 어떤 문자든 표현할 수 있다.

```text
영어: A -> [65]
한글: 가 -> [234, 176, 128]
이모지: 😊 -> [240, 159, 152, 138]
```

하지만 byte 단위만 사용하면 token sequence가 너무 길어진다. 특히 한글은 보통 한 글자가 3 bytes라서 문장이 길어질수록 token 수도 빠르게 늘어난다.

반대로 단어 단위 tokenizer를 쓰면 sequence는 짧아질 수 있지만, 처음 보는 단어를 처리하기 어렵다.

BPE는 이 둘 사이의 타협이다.

```text
byte 단위:
항상 표현 가능하지만 길다

단어 단위:
짧지만 처음 보는 단어에 약하다

BPE:
처음에는 byte에서 시작하고,
자주 나오는 조합만 점점 큰 token으로 합친다
```

## 3. 이 프로젝트의 byte-level BPE 규칙

이 프로젝트에서는 문자열을 글자 단위나 공백 단위로 먼저 자르지 않는다.

항상 다음 순서를 따른다.

```python
text.encode("utf-8")
```

즉 시작점은 문자열이 아니라 UTF-8 byte sequence다.

권장 token ID 배치는 다음과 같다.

```text
0~3:
special token
<pad>, <unk>, <bos>, <eos>

4~259:
원본 byte 0~255

260 이상:
BPE merge로 새로 만든 token
```

초기 vocabulary는 최소한 다음 정보를 가져야 한다.

```text
id_to_token[0] = "<pad>"
id_to_token[1] = "<unk>"
id_to_token[2] = "<bos>"
id_to_token[3] = "<eos>"

id_to_token[4] = bytes([0])
id_to_token[5] = bytes([1])
...
id_to_token[259] = bytes([255])
```

반대 방향 딕셔너리도 필요하다.

```text
token_to_id["<pad>"] = 0
token_to_id[bytes([0])] = 4
token_to_id[bytes([255])] = 259
```

## 4. train이 해야 하는 일

`train(corpus)`는 코퍼스를 보고 BPE merge rule과 vocabulary를 만든다.

여기서 `corpus`는 tokenizer를 학습할 긴 텍스트 묶음이다.

예:

```text
이 영화 정말 재미있다
배우 연기가 별로였다
결말이 좋았다
```

`train`의 흐름은 다음과 같다.

```text
1. vocabulary를 special token + byte token으로 초기화한다.
2. corpus를 UTF-8 byte sequence로 바꾼다.
3. 각 byte를 token ID로 바꾼다.
4. 인접한 token pair의 등장 횟수를 센다.
5. 가장 자주 등장한 pair 하나를 고른다.
6. 그 pair를 새 token ID로 vocabulary에 등록한다.
7. corpus의 token ID sequence에서 해당 pair를 새 token으로 치환한다.
8. merge rule을 self.merges에 저장한다.
9. vocab_size에 도달하거나 더 이상 합칠 pair가 없을 때까지 반복한다.
```

중요한 점은 **한 번에 여러 pair를 merge하지 않는 것**이다.

하나의 pair를 merge하면 token sequence가 바뀌고, 그 결과 다음 pair 빈도도 바뀐다. 그래서 매 반복마다 pair 빈도를 다시 계산해야 한다.

merge rule은 다음처럼 저장할 수 있다.

```python
self.merges = [
    ((238, 180), 260),
    ((260, 132), 261),
]
```

여기서 의미는 다음과 같다.

```text
(238, 180)을 260으로 합친다
(260, 132)를 261로 합친다
```

## 5. encode가 해야 하는 일

`encode(text)`는 문자열을 token ID 리스트로 바꾼다.

흐름은 다음과 같다.

```text
1. text.encode("utf-8")로 byte sequence를 만든다.
2. 각 byte를 token ID로 바꾼다.
3. train/load에서 얻은 merge rule을 학습 순서대로 적용한다.
4. add_bos_eos=True이면 앞뒤에 <bos>, <eos> ID를 붙인다.
5. 최종 token ID 리스트를 반환한다.
```

예:

```text
"가"
-> UTF-8 bytes [234, 176, 128]
-> byte token ids [238, 180, 132]
-> merge 적용 [260, 132]
-> merge 적용 [261]
```

## 6. decode가 해야 하는 일

`decode(ids)`는 token ID 리스트를 문자열로 복원한다.

흐름은 다음과 같다.

```text
1. 각 token ID가 의미하는 bytes 조각을 찾는다.
2. bytes 조각을 모두 이어 붙인다.
3. 마지막에 한 번만 UTF-8 decode를 한다.
```

중요한 점은 byte를 하나씩 decode하면 안 된다는 것이다.

한글 한 글자는 보통 3개의 byte로 이루어진다.

```text
"가" = [234, 176, 128]
```

따라서 다음처럼 전체 bytes를 먼저 합쳐야 한다.

```python
byte_seq = b"".join(byte_chunks)
text = byte_seq.decode("utf-8")
```

## 7. save와 load가 해야 하는 일

`save(path)`는 학습된 tokenizer를 파일로 저장한다.

저장해야 하는 정보는 다음과 같다.

```text
vocab_size
id_to_token
merges
```

`load(path)`는 저장된 정보를 다시 읽어서 tokenizer 상태를 복원한다.

복원 후에는 같은 문장에 대해 같은 encode 결과가 나와야 한다.

주의할 점:

```text
bytes는 JSON에 바로 저장할 수 없다.
tuple도 JSON에서는 list로 저장된다.
```

그래서 저장할 때는 bytes를 `list[int]`나 hex 문자열로 바꾸고, load할 때 다시 bytes로 복원해야 한다.

## 8. BPE가 하지 않는 일

BPE는 다음 일을 하지 않는다.

```text
단어의 의미를 이해하지 않는다.
비슷한 의미의 단어를 비슷한 벡터로 만들지 않는다.
문장을 긍정/부정으로 분류하지 않는다.
다음 token을 예측하지 않는다.
모델 가중치를 학습하지 않는다.
```

이런 일은 나중에 embedding, attention, GPT model, train loop에서 담당한다.

BPE의 책임은 다음 하나로 정리할 수 있다.

```text
텍스트와 token ID 사이를 안정적으로 변환한다.
```

즉:

```text
encode: 문자열 -> token ID 리스트
decode: token ID 리스트 -> 문자열
train: 자주 나오는 byte/token pair를 합치는 규칙 학습
save/load: 학습된 tokenizer 재사용
```
