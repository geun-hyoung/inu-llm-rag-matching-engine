"""
유틸리티 함수
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_json(file_path: str) -> Any:
    """JSON 파일을 불러옵니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    """데이터를 JSON 파일로 저장합니다."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dir(dir_path: str):
    """디렉토리가 없으면 생성합니다."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

