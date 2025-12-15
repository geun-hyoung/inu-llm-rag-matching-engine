# INU LLM RAG Matching Engine

교내 구성원을 위한 산학 매칭 알고리즘 프로젝트

## 프로젝트 개요

이 프로젝트는 교내 구성원(교수, 연구원 등)의 산학 지식 정보를 수집하고, RAG(Retrieval-Augmented Generation) 시스템을 구축하여 산학 협력 매칭을 지원하는 시스템입니다.

## 주요 기능

1. **데이터 수집 및 처리**
   - 교내 구성원의 산학 지식 정보 수집
   - 원본 데이터 정제 및 가공

2. **데이터 탐색**
   - 수집된 데이터 분석 및 통계 생성
   - 데이터 품질 검증

3. **임베딩 생성**
   - 텍스트 데이터를 벡터 임베딩으로 변환
   - RAG 시스템 구축을 위한 전처리

4. **RAG 시스템**
   - 벡터 저장소 구축
   - 의미 기반 검색 및 매칭 기능

## 프로젝트 구조

```
inu-llm-rag-matching-engine/
├── data/                    # 데이터 저장 디렉토리
│   ├── raw/                 # 원본 데이터
│   ├── processed/           # 처리된 데이터
│   ├── datasets/            # 데이터셋
│   └── rag_store/           # RAG 벡터 저장소
├── data_collection/         # 데이터 수집 모듈
│   ├── __init__.py
│   └── collector.py         # 데이터 수집기
├── data_processing/         # 데이터 처리 모듈
│   ├── __init__.py
│   └── processor.py         # 데이터 처리기
├── data_exploration/        # 데이터 탐색 모듈
│   ├── __init__.py
│   └── explorer.py          # 데이터 탐색기
├── embedding/               # 임베딩 모듈
│   ├── __init__.py
│   └── embedder.py          # 임베딩 생성기
├── rag_system/              # RAG 시스템 모듈
│   ├── __init__.py
│   ├── vector_store.py      # 벡터 저장소
│   └── rag_engine.py        # RAG 엔진
├── models/                  # 모델 관련
│   ├── __init__.py
│   └── model_config.py      # 모델 설정
├── utils/                   # 유틸리티
│   ├── __init__.py
│   └── helpers.py           # 헬퍼 함수
├── config/                  # 설정 파일
│   ├── __init__.py
│   └── settings.py          # 프로젝트 설정
├── results/                 # 결과 저장 디렉토리
│   └── save_results.py      # 결과 저장 유틸리티
├── .gitignore
└── README.md
```

## 사용 방법

### 1. 데이터 수집
```python
from data_collection.collector import DataCollector

collector = DataCollector()
data = [...]  # 수집할 데이터
collector.collect_data("faculty_info", data)
```

### 2. 데이터 처리
```python
from data_processing.processor import DataProcessor

processor = DataProcessor()
processor.process_data("faculty_info")
```

### 3. 데이터 탐색
```python
from data_exploration.explorer import DataExplorer

explorer = DataExplorer()
explorer.save_exploration_results("faculty_info")
```

### 4. 임베딩 생성
```python
from embedding.embedder import Embedder

embedder = Embedder()
# 처리된 데이터 로드 후 임베딩 생성
```

### 5. RAG 시스템 사용
```python
from rag_system.rag_engine import RAGEngine
from rag_system.vector_store import VectorStore
from embedding.embedder import Embedder

store = VectorStore()
embedder = Embedder()
rag = RAGEngine(store, embedder)

# 문서 추가
rag.add_documents(documents)

# 검색
results = rag.query("인공지능 전문가", top_k=5)
```

## 결과 저장

```python
from results.save_results import ResultsSaver

saver = ResultsSaver()
saver.save_with_timestamp(result_data, "experiment_result")
```

## 개발 환경

- Python 3.8+
- 필요한 패키지는 `requirements.txt` 참조

## 라이선스

이 프로젝트는 교내 연구 목적으로 사용됩니다.
