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
   - Hybrid retrieval (Local + Global search)
   - Graph-based knowledge expansion

4. **AHP-based Ranking System** 🆕
   - Professor document aggregation by data type
   - AHP (Analytic Hierarchy Process) algorithm
   - Multi-criteria professor ranking
   - Configurable weights for patent/article/project

5. **Report Generation** 🆕
   - Industry-academia matching recommendation reports
   - Multiple output formats (JSON, PDF, HTML)
   - Detailed professor information and matching rationale

## Project Structure

```
inu-llm-rag-matching-engine/
├── data/                    # Data storage directory
│   ├── article/             # Article data
│   ├── patent/              # Patent data
│   ├── project/              # Project data
│   ├── processed/           # Processed data (embeddings)
│   ├── rag_store/           # RAG vector store
│   ├── test/                # Test data (filtered)
│   └── train/               # Training data (filtered)
├── data_collection/         # Data collection scripts
│   ├── article_collection.py
│   ├── patent_collection.py
│   └── project_collection.py
├── data_exploration/        # Data exploration scripts
│   ├── patent_eda.py
│   ├── article_eda.py
│   └── project_eda.py
├── data_filtering/          # Data filtering and preprocessing
│   ├── article_filtering.py
│   ├── patent_filtering.py
│   ├── project_filtering.py
│   └── text_preprocessing.py
├── src/                     # Core source code
│   ├── rag/                 # RAG system modules
│   │   ├── embedding/       # Embedding modules
│   │   ├── index/           # Entity extraction
│   │   ├── preprocessing/   # Text preprocessing
│   │   ├── query/           # Retrieval modules
│   │   ├── store/           # Vector & Graph stores
│   │   └── prompts.py       # LLM prompts
│   ├── ranking/             # 🆕 AHP-based ranking system
│   │   ├── professor_aggregator.py  # Professor document aggregation
│   │   ├── ahp.py           # AHP algorithm implementation
│   │   └── ranker.py        # Professor ranking
│   ├── reporting/           # 🆕 Report generation
│   │   ├── report_generator.py
│   │   └── templates/       # Report templates
│   └── evaluation/          # Evaluation metrics
│       ├── metrics.py
│       └── noise_rate.py
├── scripts/                 # Execution scripts
│   ├── build_index.py       # Index building pipeline
│   ├── query.py             # Simple query execution
│   ├── match.py             # 🆕 Full matching pipeline (RAG + AHP + Report)
│   └── run_evaluation.py    # Evaluation execution
├── config/                  # Configuration files
│   ├── database.py          # Database connection settings
│   ├── settings.py          # Project settings
│   └── ahp_config.py        # 🆕 AHP weights and configuration
├── results/                 # Analysis results
│   ├── eda/                 # EDA results
│   └── reports/             # 🆕 Generated reports
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
python data_exploration/article_eda.py
python data_exploration/project_eda.py
```

### RAG System (Simple Query)
```bash
python scripts/query.py "딥러닝 의료영상 전문가" --doc-types patent article project
```

### Full Matching Pipeline 🆕
```bash
# RAG 검색 → 교수 집계 → AHP 랭킹 → 보고서 생성
python scripts/match.py "딥러닝 의료영상 전문가" \
    --doc-types patent article project \
    --top-n 10 \
    --output-format json
```

### Index Building
```bash
python scripts/build_index.py --doc-type patent
python scripts/build_index.py --doc-type article
python scripts/build_index.py --doc-type project
```

### Evaluation
```bash
python scripts/run_evaluation.py --retriever hybrid
```

## Workflow

### Complete Pipeline Flow

1. **Data Collection** → Collect patent/article/project data with professor information
2. **Data Processing** → Filter and preprocess data
3. **Index Building** → Extract entities/relations and build vector/graph stores
4. **Query Processing** → User query → RAG retrieval (3 data types)
5. **Professor Aggregation** → Aggregate documents by professor for each data type
6. **AHP Ranking** → Calculate professor scores using AHP algorithm
7. **Report Generation** → Generate matching recommendation report

### Key Features

- **Hybrid Retrieval**: Combines local (entity-based) and global (relation-based) search
- **Multi-type Support**: Handles patent, article, and project data simultaneously
- **AHP-based Ranking**: Uses Analytic Hierarchy Process for multi-criteria decision making
- **Professor-centric**: All documents are mapped to professors for matching

## Development Environment

- Python 3.11.9
- See `requirements.txt` for required packages

## Recent Updates 🆕

- Added AHP-based ranking system (`src/ranking/`)
- Added report generation module (`src/reporting/`)
- Added integrated matching pipeline (`scripts/match.py`)
- Enhanced EDA with abstract analysis and visualization
- Added AHP configuration (`config/ahp_config.py`)

## License

This project is for internal research purposes.
