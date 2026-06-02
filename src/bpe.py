# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    이 클래스의 역할은 "문자열을 모델이 다룰 수 있는 token ID 리스트로 바꾸고,
    token ID 리스트를 다시 문자열로 복원하는 것"입니다. 여기서는 단어 의미를
    학습하지 않습니다. 의미 벡터는 나중에 embedding/model 학습에서 만들어집니다.

    BPE가 학습하는 것은 "자주 붙어 나오는 byte/token 쌍을 하나의 새 token으로
    합치는 규칙"입니다. 예를 들어 (238, 180)이 자주 나오면 새 token 260으로
    합치는 rule을 만들고, 이후 encode에서도 같은 rule을 적용합니다.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.

        이 함수가 끝나면 최소 vocabulary가 준비되어야 합니다.
        이 최소 vocabulary만 있어도 모든 UTF-8 문자열을 byte 단위로 encode/decode할 수 있습니다.

        필요한 딕셔너리 구조:
        - self.id_to_token[token_id] = token 문자열 또는 bytes 조각
        - self.token_to_id[token 문자열 또는 bytes 조각] = token_id
        """
        self.id_to_token.clear()
        self.token_to_id.clear()
        self.merges.clear()

        self.token_to_id.update(SPECIAL_IDS)
        for i in range(len(SPECIAL_TOKENS)):
            self.id_to_token[i] = SPECIAL_TOKENS[i]

        # 모든 UTF-8 문자열은 0~255 범위의 byte 조합으로 표현됩니다.
        for i in range(256):
            tokenid = i + 4
            token_bytes = bytes([i])
            self.id_to_token[tokenid] = token_bytes
            self.token_to_id[token_bytes] = tokenid

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        train의 목표:
        - 코퍼스 문자열을 보고 자주 등장하는 이웃 token pair를 찾습니다.
        - 가장 자주 등장한 pair 하나를 새 token으로 합칩니다.
        - 이 과정을 반복해서 vocabulary를 self.vocab_size까지 늘립니다.

        용어:
        - corpus: tokenizer 학습에 사용할 긴 텍스트 묶음
        - ids: corpus를 현재 token ID 리스트로 표현한 작업용 리스트
        - pair: ids 안에서 서로 붙어 있는 두 token ID, 예: (238, 180)
        - merge rule: 어떤 pair를 어떤 새 token ID로 합쳤는지 기록한 규칙

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.

        구현 순서:
        1. `_init_special_tokens()`로 vocabulary를 초기 상태로 만듭니다.
        2. `corpus.encode("utf-8")` 결과의 각 byte b를 token ID `BYTE_OFFSET + b`로 바꿉니다.
        3. 현재 ids에서 인접 pair 등장 횟수를 딕셔너리로 셉니다.
           예: pair_counts[(ids[i], ids[i + 1])] += 1
        4. 가장 많이 등장한 pair 하나를 고릅니다.
        5. 새 token ID를 만들고, 두 token의 bytes를 이어 붙여 vocab에 등록합니다.
        6. `self.merges.append((best_pair, new_id))`처럼 merge rule을 순서대로 저장합니다.
        7. ids 안의 best_pair를 new_id 하나로 치환합니다.
        8. 목표 vocab size에 도달하거나 더 이상 합칠 pair가 없을 때까지 반복합니다.

        주의:
        - pair 빈도를 한 번 세고 상위 여러 개를 한꺼번에 merge하지 않습니다.
          하나를 merge하면 ids가 바뀌므로, 다음 반복에서 pair 빈도를 다시 세야 합니다.
        - merge rule에는 pair의 빈도수가 아니라 새 token ID를 저장해야 합니다.
        - `self.merges`는 딕셔너리보다 리스트가 좋습니다. encode에서 학습 순서대로 적용해야 하기 때문입니다.
        """
        def count_pairs(ids):
            pair_dic = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_dic[pair] = pair_dic.get(pair, 0) + 1
            return pair_dic

        def updated_ids(ids, best_pair, new_id):
            new_ids = []
            i = 0

            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            return new_ids

        self._init_special_tokens()

        byted_corpus = corpus.encode("utf-8")
        ids = [BYTE_OFFSET + b for b in byted_corpus]

        while len(self.id_to_token) < self.vocab_size:
            pair_counts = count_pairs(ids)

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)

            if (pair_counts[best_pair]) == 1:
                break

            left_id, right_id = best_pair
            new_token = self._token_to_bytes(left_id) + self._token_to_bytes(right_id)
            token_id = len(self.id_to_token)

            self.id_to_token[token_id] = new_token
            self.token_to_id[new_token] = token_id
            self.merges.append((best_pair, token_id))

            ids = updated_ids(ids, best_pair, token_id)

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.

        저장해야 하는 정보:
        - vocab_size
        - id_to_token: token ID가 어떤 문자열/bytes 조각을 뜻하는지
        - merges: train에서 만든 merge rule 목록

        주의:
        - JSON key는 문자열이어야 하므로 token ID를 저장할 때 문자열 key로 바뀔 수 있습니다.
        - bytes는 JSON에 바로 저장할 수 없으므로 list[int]나 hex 문자열 같은 형태로 바꿔 저장해야 합니다.
        - tuple pair도 JSON에서는 list로 저장되므로 load에서 다시 tuple로 복원해야 합니다.
        """
        serializable_vocab = {}

        for token_id, token in self.id_to_token.items():
            if isinstance(token, str):
                serializable_vocab[token_id] = {
                    "type": "str",
                    "value": token,
                }
            elif isinstance(token, bytes):
                serializable_vocab[token_id] = {
                    "type": "bytes",
                    "value": list(token),
                }
            elif isinstance(token, tuple):
                serializable_vocab[token_id] = {
                    "type": "tuple",
                    "value": list(token),
                }
            else:
                raise TypeError(f"Unsupported token type: {type(token)}")

        serializable_merges = [self._serialize_merge(merge) for merge in self.merges]

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": serializable_vocab,
            "merges": serializable_merges,
        }

        path = Path(path)
        with path.open('w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.

        load의 목표:
        - save에서 저장한 vocab과 merge rule을 그대로 복원합니다.
        - load 후에는 같은 text에 대해 save 전 tokenizer와 같은 encode 결과가 나와야 합니다.

        복원해야 하는 필드:
        - self.vocab_size
        - self.id_to_token
        - self.token_to_id
        - self.merges
        """

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.id_to_token.clear()
        self.token_to_id.clear()
        self.merges.clear()

        # list로 저장되있는 것을 원래 type으로 변환해야 함
        for token_id_str, entity in data["id_to_token"].items(): # key, value
            if entity["type"] == "str":
                value = entity["value"]
            elif entity["type"] == "bytes":
                value = bytes(entity["value"])
            elif entity["type"] == "tuple":
                value = tuple(entity["value"])
            else:
                raise ValueError(f"Unsupported token type: {entity['type']}")
            
            token_id = int(token_id_str)
            self.id_to_token[token_id] = value
            self.token_to_id[value] = token_id

        for entity in data["merges"]:
            self.merges.append(self._deserialize_merge(entity))

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.

        구현 순서:
        1. text.encode("utf-8")로 byte sequence를 만듭니다.
        2. 각 byte b를 token ID `BYTE_OFFSET + b`로 바꿉니다.
        3. self.merges에 저장된 rule을 앞에서부터 순서대로 적용합니다.
           예: ((238, 180), 260)이 있으면 ids 안의 238, 180을 260 하나로 바꿉니다.
        4. add_bos_eos가 True이면 맨 앞에 BOS, 맨 뒤에 EOS token ID를 붙입니다.
        5. 최종 ids를 반환합니다.
        """
        ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]

        for merge in self.merges:
            target_pair, merged_id = self._normalize_merge_rule(merge)
            if merged_id is None:
                continue

            new_ids = []
            i = 0

            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == target_pair:
                    new_ids.append(merged_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids

        if add_bos_eos:
            ids = [self.get_bos_id(), *ids, self.get_eos_id()]

        return ids
    
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.

        구현 순서:
        1. ids를 앞에서부터 순회합니다.
        2. special token ID를 만나면 skip_special 값에 따라 건너뛰거나 문자열로 처리합니다.
        3. 일반 token ID는 self.id_to_token에서 bytes 조각을 찾습니다.
        4. 모든 bytes 조각을 하나로 이어 붙입니다.
        5. 마지막에 한 번만 `.decode("utf-8")`를 호출해서 문자열로 복원합니다.

        왜 마지막에 한 번만 decode하나:
        - 한글 한 글자는 보통 3개의 byte로 이루어집니다.
        - byte 하나씩 decode하면 UTF-8 문자 하나가 완성되지 않아 오류가 날 수 있습니다.
        """
        special_ids = set(SPECIAL_IDS.values())
        byte_chunks = []
        text_chunks = []

        for token_id in ids:
            if token_id in special_ids:
                if skip_special:
                    continue
                if byte_chunks:
                    text_chunks.append(b"".join(byte_chunks).decode("utf-8", errors="replace"))
                    byte_chunks = []
                text_chunks.append(self.id_to_token[token_id])
                continue

            byte_chunks.append(self._token_to_bytes(token_id))

        if byte_chunks:
            text_chunks.append(b"".join(byte_chunks).decode("utf-8", errors="replace"))

        return "".join(text_chunks)

    def _token_to_bytes(self, token_id: int) -> bytes:
        """token_id가 뜻하는 원본 byte 조각을 반환합니다."""
        token = self.id_to_token[token_id]

        if isinstance(token, bytes):
            return token

        if isinstance(token, tuple):
            return b"".join(self._token_to_bytes(child_id) for child_id in token)

        if isinstance(token, str):
            return token.encode("utf-8")

        raise TypeError(f"Unsupported token type: {type(token)}")

    def _normalize_merge_rule(self, merge):
        """merge rule을 ((left, right), new_id) 형태로 맞춥니다."""
        if (
            isinstance(merge, (list, tuple))
            and len(merge) == 2
            and isinstance(merge[0], (list, tuple))
        ):
            return tuple(merge[0]), int(merge[1])
        return tuple(merge), None

    def _serialize_merge(self, merge):
        pair, token_id = self._normalize_merge_rule(merge)
        if token_id is None:
            return {
                "type": "pair",
                "value": list(pair),
            }
        return {
            "type": "rule",
            "pair": list(pair),
            "id": token_id,
        }

    def _deserialize_merge(self, entity):
        if entity["type"] == "pair":
            return tuple(entity["value"])
        if entity["type"] == "rule":
            return (tuple(entity["pair"]), entity["id"])
        if entity["type"] == "tuple":
            return (tuple(entity["value"]), entity["id"])
        raise ValueError(f"Unsupported merge type: {entity['type']}")
