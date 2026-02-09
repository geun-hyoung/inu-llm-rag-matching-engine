# INU LLM RAG Matching Engine

RAG + AHP로 **검색어에 맞는 교수·논문·특허·연구과제**를 추천하고, **PDF 보고서**까지 생성하는 파이프라인입니다.

---

## Quick Start

```bash
# 환경
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m playwright install chromium

# 설정: config/settings.py (API 키 등)

# 인덱스 구축 (최초 1회)
python scripts/build_index.py --doc-type patent
python scripts/build_index.py --doc-type article
python scripts/build_index.py --doc-type project

# 웹 앱 실행 (검색 → 추천 → 보고서 생성)
streamlit run scripts/app.py
```

---

## 주요 명령어

| 용도 | 명령어 |
|------|--------|
| 인덱스 구축 | `python scripts/build_index.py --doc-type patent` (article, project 동일) |
| 검색 + 랭킹 + 보고서 | `python scripts/match.py "검색어" --doc-types patent article project --top-n 10` |
| 웹 UI | `streamlit run scripts/app.py` |
| AHP만 실행 | `python scripts/run_ahp.py` |

---

## 구조

- **config/** — 설정, AHP 가중치
- **data_collection/** — 특허·논문·연구과제 수집
- **scripts/** — `build_index`, `match`, `app`(Streamlit)
- **src/rag/** — 임베딩, 벡터/그래프 저장소, 검색
- **src/ranking/** — 교수 집계, AHP 랭킹
- **src/reporting/** — 보고서 생성(JSON/텍스트/PDF)
- **results/** — 실행 결과, 보고서 PDF

---

## 기술

- Python 3.11+
- RAG: OpenAI 임베딩, ChromaDB, 그래프 저장소
- 랭킹: AHP (`config/ahp_config.py`)
- PDF: Playwright(Chromium)

---

*Internal research use.*
