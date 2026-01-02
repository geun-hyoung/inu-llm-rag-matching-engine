# INU LLM RAG Matching Engine

교내 구성원을 위한 산학 매칭 알고리즘 프로젝트

## 프로젝트 개요

이 프로젝트는 교내 구성원(교수, 연구원 등)의 산학 지식 정보를 수집하는 시스템입니다.
KIPRIS API를 통해 특허 데이터를 수집하고, 교수 정보와 함께 저장합니다.

## 주요 기능

1. **KIPRIS 특허 데이터 수집**
   - MariaDB의 `tb_inu_tech` 테이블에서 특허 출원번호 조회
   - `v_emp1` 테이블과 조인하여 교수 정보 매칭
   - KIPRIS API를 통해 특허 상세 정보 수집
   - JSON 파일로 저장

## 프로젝트 구조

```
inu-llm-rag-matching-engine/
├── data/                    # 데이터 저장 디렉토리
│   ├── raw/                 # 원본 데이터
│   │   ├── kipris_data.json              # KIPRIS 특허 데이터
│   │   └── kipris_professor_info.json    # 교수 정보
│   ├── processed/           # 처리된 데이터
│   └── datasets/            # 데이터셋
├── data_collection/         # 데이터 수집 모듈
│   ├── __init__.py
│   └── kipris_collector.py  # KIPRIS 특허 데이터 수집기
├── config/                  # 설정 파일
│   ├── database.py          # 데이터베이스 연결 설정
│   └── settings.py          # 프로젝트 설정 (API 키 등)
├── .gitignore
├── requirements.txt
└── README.md
```

## 사용 방법

### KIPRIS 특허 데이터 수집

```python
from data_collection.kipris_collector import KIPRISCollector
from config.settings import KIPRIS_API_KEY

# 수집기 생성
collector = KIPRISCollector(api_key=KIPRIS_API_KEY)

# JSON 파일로 저장 (limit=None이면 전체 수집, 호출 제한까지)
collector.collect_and_save(limit=None)
```

## 수집되는 데이터

### 특허 데이터 (`kipris_data.json`)
- `tech_aplct_id`: 특허 출원번호
- `inpt_mbr_id`: 교수 사번
- `kipris_index_no`: 인덱스 번호
- `kipris_register_status`: 등록 상태
- `kipris_application_date`: 출원일
- `kipris_abstract`: 특허 요약
- `kipris_application_name`: 발명의 명칭
- `professor_info`: 교수 정보 (v_emp1 테이블에서 가져온 모든 정보)

### 교수 정보 (`kipris_professor_info.json`)
- `professor_info`: 교수 정보 (중복 제거)
- `collected_from_kipris`: KIPRIS에서 수집됨 표시

## 개발 환경

- Python 3.11
- 필요한 패키지는 `requirements.txt` 참조

## 라이선스

이 프로젝트는 교내 연구 목적으로 사용됩니다.
