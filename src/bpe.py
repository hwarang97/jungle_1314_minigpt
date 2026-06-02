# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json
from collections import Counter


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

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        # 흐름(의사코드):
        # 1. 목표 vocabulary 크기를 저장합니다.
        # 2. id->token, token->id 사전을 준비합니다.
        # 3. BPE merge rule 목록을 비워 둡니다.
        self.vocab_size = vocab_size
        # id_to_token: 모델 출력 token id를 다시 token 객체(bytes, pair, special str)로 되돌릴 때 사용합니다.
        self.id_to_token = {}
        # token_to_id: encode 과정에서 token 객체를 정수 id로 바꿀 때 사용합니다.
        self.token_to_id = {}
        # merges는 BPE가 학습한 pair 병합 순서입니다. encode 때도 반드시 이 순서대로 적용합니다.
        self.merges = []

    def _init_special_tokens(self):
        """
        특수 토큰 4개를 고정 ID 0~3에 등록하고,
        byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.

        흐름(의사코드):
        1. 기존 vocabulary 사전을 비웁니다.
        2. special token 4개를 0~3에 넣습니다.
        3. byte token 256개를 4~259에 넣습니다.
        """
        self.id_to_token = {}
        self.token_to_id = {}

        # ('책내용') 2장: 텍스트는 모델이 바로 계산할 수 없으므로 먼저 고정된 token id로 바꾼다.
        for i, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[i] = token
            self.token_to_id[token] = i

        # ('책내용') 과제 tokenizer는 tiktoken을 쓰지 않고 UTF-8 byte를 기본 어휘로 직접 둔다.
        for i in range(NUM_BYTES):
            byte_token = bytes([i])
            self.id_to_token[BYTE_OFFSET + i] = byte_token
            self.token_to_id[byte_token] = BYTE_OFFSET + i

    def get_pad_id(self):
        """padding 토큰 ID. 흐름: 고정 special id 0을 반환합니다."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID. 흐름: 고정 special id 1을 반환합니다."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID. 흐름: 고정 special id 2를 반환합니다."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID. 흐름: 고정 special id 3을 반환합니다."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.

        흐름(의사코드):
        1. special token과 byte token으로 vocabulary를 초기화합니다.
        2. corpus를 UTF-8 byte token id sequence로 바꿉니다.
        3. 가장 자주 등장하는 이웃 pair를 찾습니다.
        4. 해당 pair를 새 token id로 등록합니다.
        5. sequence 안의 pair를 새 token으로 치환합니다.
        6. vocab_size에 도달하거나 반복할 pair가 없으면 종료합니다.
        """
        self._init_special_tokens()
        self.merges = []
        # UTF-8 byte 값 0~255를 특수 토큰 4개 뒤의 id 범위(4~259)로 옮깁니다.
        # 예: byte 65("A") -> token id 69
        token_ids = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size and len(token_ids) >= 2:
            # 현재 token sequence에서 이웃한 두 token이 얼마나 자주 붙어 나오는지 셉니다.
            pair_counts = Counter(zip(token_ids, token_ids[1:]))
            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]
            # 한 번만 나온 pair까지 merge하면 vocabulary만 커지고 일반화 이득은 거의 없습니다.
            if count < 2:
                break

            new_id = len(self.id_to_token)
            self.merges.append(best_pair)
            # merge token은 바로 bytes가 아니라 "왼쪽 token id, 오른쪽 token id" pair로 저장합니다.
            # decode 때 이 pair를 재귀적으로 펼쳐 원래 byte까지 복원합니다.
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id

            # corpus token sequence 안의 best_pair를 새 token id 하나로 실제 치환합니다.
            token_ids = self._merge_pair(token_ids, best_pair, new_id)

    def save(self, path: str | Path):
        """
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.

        흐름(의사코드):
        1. 저장 경로의 폴더를 준비합니다.
        2. bytes/tuple token을 JSON 가능한 dict로 바꿉니다.
        3. vocab_size, id_to_token, merges를 JSON으로 저장합니다.
        """
        path = Path(path)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "vocab_size": self.vocab_size,
            # JSON object key는 문자열이어야 하므로 token id를 str로 바꾸어 저장합니다.
            "id_to_token": {
                str(idx): self._serialize_token(token)
                for idx, token in self.id_to_token.items()
            },
            # tuple pair도 JSON에는 list로 저장하고, load에서 다시 tuple로 복원합니다.
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path):
        """
        save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.

        흐름(의사코드):
        1. JSON 파일을 읽습니다.
        2. merges를 tuple pair로 복원합니다.
        3. id_to_token을 원래 token 객체로 복원합니다.
        4. encode에 필요한 token_to_id 역방향 사전을 다시 만듭니다.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab_size = payload.get("vocab_size", self.vocab_size)
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = [tuple(pair) for pair in payload.get("merges", [])]

        for idx_text, token_payload in payload["id_to_token"].items():
            idx = int(idx_text)
            token = self._deserialize_token(token_payload)
            self.id_to_token[idx] = token
            # load 후 encode가 바로 동작하려면 역방향 사전도 반드시 다시 만들어야 합니다.
            self.token_to_id[token] = idx

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.

        흐름(의사코드):
        1. tokenizer가 비어 있으면 byte vocabulary를 초기화합니다.
        2. text를 UTF-8 byte token id 목록으로 바꿉니다.
        3. 학습한 merge rule을 순서대로 적용합니다.
        4. 옵션에 따라 BOS/EOS를 앞뒤에 붙입니다.
        5. token id 목록을 반환합니다.
        """
        if not self.id_to_token:
            self._init_special_tokens()

        # ('책내용') 2장: token id는 의미가 담긴 숫자가 아니라 embedding에 넣기 전의 번호표다.
        token_ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]
        for pair in self.merges:
            new_id = self.token_to_id.get(pair)
            if new_id is not None:
                # 학습 때 만든 merge rule을 순서대로 적용해야 train 때의 vocabulary 의미와 맞습니다.
                token_ids = self._merge_pair(token_ids, pair, new_id)

        if add_bos_eos:
            token_ids = [self.get_bos_id()] + token_ids + [self.get_eos_id()]

        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.

        흐름(의사코드):
        1. token id 하나씩 원본 byte로 펼칩니다.
        2. special token은 skip_special이면 건너뜁니다.
        3. 모은 bytearray를 UTF-8 문자열로 한 번에 decode합니다.
        """
        byte_values = bytearray()
        for token_id in ids:
            # merge token은 pair 안에 pair가 또 있을 수 있으므로 원본 byte까지 재귀적으로 펼칩니다.
            byte_values.extend(self._token_id_to_bytes(token_id, skip_special))
        # 한글은 여러 byte가 모여 한 글자가 되므로 마지막에 한 번에 decode해야 깨지지 않습니다.
        return bytes(byte_values).decode("utf-8", errors="replace")

    @staticmethod
    def _merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        # 흐름(의사코드):
        # 1. token sequence를 왼쪽부터 훑습니다.
        # 2. 현재 token과 다음 token이 pair와 같으면 new_id 하나로 바꿉니다.
        # 3. pair가 아니면 현재 token을 그대로 둡니다.
        merged = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                # pair의 두 token을 새 token 하나로 압축하고, 두 칸을 소비합니다.
                merged.append(new_id)
                i += 2
            else:
                # pair가 아니면 기존 token을 그대로 유지합니다.
                merged.append(token_ids[i])
                i += 1
        return merged

    @staticmethod
    def _serialize_token(token):
        # 흐름(의사코드):
        # 1. bytes token은 byte 값 list로 바꿉니다.
        # 2. pair token은 token id list로 바꿉니다.
        # 3. special string token은 문자열 그대로 저장합니다.
        if isinstance(token, bytes):
            return {"type": "bytes", "value": list(token)}
        if isinstance(token, tuple):
            return {"type": "pair", "value": list(token)}
        return {"type": "str", "value": token}

    @staticmethod
    def _deserialize_token(payload):
        # 흐름(의사코드):
        # 1. 저장된 type을 확인합니다.
        # 2. bytes면 bytes 객체로, pair면 tuple로, str이면 문자열로 복원합니다.
        token_type = payload["type"]
        value = payload["value"]
        if token_type == "bytes":
            return bytes(value)
        if token_type == "pair":
            return tuple(value)
        return value

    def _token_id_to_bytes(self, token_id: int, skip_special: bool) -> bytes:
        # 흐름(의사코드):
        # 1. token id로 token 객체를 찾습니다.
        # 2. bytes token이면 그대로 반환합니다.
        # 3. pair token이면 양쪽 token을 재귀적으로 byte까지 펼칩니다.
        # 4. special token이면 skip 여부에 따라 버리거나 문자열 byte로 바꿉니다.
        token = self.id_to_token.get(token_id)
        if token is None:
            return b"" if skip_special else UNK_TOKEN.encode("utf-8")

        if isinstance(token, bytes):
            return token
        if isinstance(token, tuple):
            left, right = token
            # 새 BPE token은 결국 byte token들의 묶음이므로 양쪽을 끝까지 펼쳐 붙입니다.
            return self._token_id_to_bytes(left, skip_special) + self._token_id_to_bytes(right, skip_special)

        if token in SPECIAL_TOKENS and skip_special:
            return b""
        return str(token).encode("utf-8")
