# INU LLM RAG Matching Engine

Industry-Academia Matching Algorithm Project for University Members

## Project Overview

This project collects industry-academia knowledge information for university members (professors, researchers, etc.), performs exploratory data analysis (EDA), and implements a RAG (Retrieval-Augmented Generation) system for matching.

## Features

1. **Data Collection**
   - Patent data collection through KIPRIS API
   - Article data collection through EBSCO
   - Data is stored with professor information

2. **Data Exploration**
   - Exploratory Data Analysis (EDA) for patent data
   - Analysis focused on professor-patent relationships
   - Results saved in JSON format

3. **RAG System**
   - Text embedding generation
   - Vector store for similarity search
   - RAG engine for retrieval-augmented generation

## Project Structure

```
inu-llm-rag-matching-engine/
├── data/                    # Data storage directory
│   ├── article/             # Article data
│   ├── patent/              # Patent data
│   ├── processed/           # Processed data (embeddings)
│   └── rag_store/           # RAG vector store
├── data_collection/         # Data collection scripts
│   ├── article_collection.py
│   └── patent_collection.py
├── data_exploration/        # Data exploration scripts
│   └── patent_eda.py
├── src/                     # Core source code
│   ├── rag/                 # RAG system modules
│   │   ├── engine.py        # RAG engine
│   │   ├── vector_store.py  # Vector store
│   │   └── retriever.py     # Retriever
│   ├── embedding/           # Embedding modules
│   │   ├── encoder.py       # Text encoder
│   │   └── model.py         # Embedding model
│   └── utils/               # Utility functions
│       └── helpers.py
├── config/                  # Configuration files
│   ├── database.py          # Database connection settings
│   └── settings.py          # Project settings
├── results/                 # Analysis results
│   └── eda/                 # EDA results
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m playwright install  # Install Playwright browsers
```

3. Configure settings:
   - Set database credentials in `config/database.py`
   - Set API keys and model settings in `config/settings.py`

## Usage

### Patent Data Collection
```bash
python data_collection/patent_collection.py
```

### Article Data Collection
```bash
python data_collection/article_collection.py
```

### Data Exploration
```bash
python data_exploration/patent_eda.py
```

### RAG System
```python
from src.rag.engine import RAGEngine
from src.rag.vector_store import VectorStore
from src.embedding.encoder import Embedder

# Initialize components
store = VectorStore()
embedder = Embedder()
rag = RAGEngine(store, embedder)

# Add documents
documents = [...]
rag.add_documents(documents)

# Query
results = rag.query("search query", top_k=5)
```

## Development Environment

- Python 3.11
- See `requirements.txt` for required packages

## License

This project is for internal research purposes.
