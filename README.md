# INU LLM RAG Matching Engine

Industry-Academia Matching Algorithm Project for University Members

## Project Overview

This project collects industry-academia knowledge information for university members (professors, researchers, etc.) and performs exploratory data analysis (EDA) on the collected data.

## Features

1. **Data Collection**
   - Patent data collection through KIPRIS API
   - Article data collection through EBSCO
   - Data is stored with professor information

2. **Data Exploration**
   - Exploratory Data Analysis (EDA) for patent data
   - Analysis focused on professor-patent relationships
   - Results saved in JSON format

## Project Structure

```
inu-llm-rag-matching-engine/
├── data/                    # Data storage directory
│   ├── article/             # Article data
│   └── patent/              # Patent data
├── data_collection/         # Data collection module
│   ├── article_collection.py
│   └── patent_collection.py
├── data_exploration/        # Data exploration module
│   └── patent_eda.py
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
   - Set API keys in `config/settings.py`

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

## Development Environment

- Python 3.11
- See `requirements.txt` for required packages

## License

This project is for internal research purposes.
