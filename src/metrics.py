# -*- coding: utf-8 -*-
"""학습/평가 기록을 JSONL 파일로 남기는 작은 유틸리티."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def append_jsonl_record(path: str | Path | None, record: dict[str, Any]) -> None:
    """
    record 하나를 JSONL 파일 끝에 추가합니다.

    흐름(의사코드):
    1. path가 None이면 기록을 건너뜁니다.
    2. logs 폴더 같은 parent directory를 만듭니다.
    3. 기록 시각을 created_at으로 붙입니다.
    4. JSON 한 줄로 append합니다.
    """
    if path is None:
        return

    path_obj = Path(path)
    if path_obj.parent != Path("."):
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
