# INU LLM RAG Matching Engine

**Match industry needs with university researchers** using RAG (Retrieval-Augmented Generation) and AHP ranking.

---

## What it does

1. **Collect** — Patents, articles, and research projects (with professor links).
2. **Index** — Build vector + graph stores for semantic search.
3. **Query** — Natural-language query → RAG retrieves relevant docs by type (patent/article/project).
4. **Rank** — AHP weights (time, contribution, scale, status) → ranked professor list.
5. **Report** — JSON, HTML, or PDF recommendation report.

---

## Quick start

```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m playwright install chromium

# 2. Set config (API keys, DB, etc.) in config/
# 3. Build index (once)
python scripts/build_index.py --doc-type patent
python scripts/build_index.py --doc-type article
python scripts/build_index.py --doc-type project

# 4. Run matching
python scripts/match.py "your search query" --doc-types patent article project --top-n 10

# 5. Or use the web UI
streamlit run scripts/app.py
```

---

## Main commands

| Task | Command |
|------|--------|
| Build index | `python scripts/build_index.py --doc-type <patent\|article\|project>` |
| Simple RAG query | `python scripts/query.py "query" --doc-types patent article project` |
| Full match + report | `python scripts/match.py "query" --doc-types patent article project --top-n 10` |
| AHP ranking only | `python scripts/run_ahp.py` |
| Web app | `streamlit run scripts/app.py` |
| EDA | `python data_exploration/patent_eda.py` (and article_eda, project_eda) |

---

## Project layout

```
├── config/           # Settings, AHP weights (ahp_config.py)
├── data_collection/  # Fetch patent, article, project data
├── data_filtering/   # Filter & preprocess text
├── data_exploration/ # EDA scripts
├── scripts/          # build_index, query, match, app (Streamlit), run_ahp
├── src/
│   ├── rag/          # Embedding, vector/graph store, retrieval
│   ├── ranking/      # Professor aggregation + AHP ranker
│   ├── reporting/    # Report generator (JSON/HTML/PDF)
│   └── evaluation/   # Metrics, noise rate
├── results/          # EDA outputs, reports, run logs
└── requirements.txt
```

---

## Tech

- **Python 3.11+**
- **RAG**: OpenAI embeddings, ChromaDB, LightRAG-style graph
- **Ranking**: AHP (configurable weights in `config/ahp_config.py`)
- **Report PDF**: Playwright (Chromium)

---

## License

Internal research use.
