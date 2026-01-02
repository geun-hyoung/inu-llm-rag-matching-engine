# INU LLM RAG Matching Engine

Industry-Academia Matching Algorithm Project for University Members

## Project Overview

This project collects industry-academia knowledge information for university members (professors, researchers, etc.).
It collects patent data through the KIPRIS API and stores it along with professor information.

## Features

1. **KIPRIS Patent Data Collection**
   - Query patent application numbers from MariaDB `tb_inu_tech` table
   - Match professor information by joining with `v_emp1` table
   - Collect detailed patent information through KIPRIS API
   - Save to JSON file

## Project Structure

```
inu-llm-rag-matching-engine/
├── data/                    # Data storage directory
│   └── patent/              # Patent data
│       └── kipris_data.json # Collected patent data with professor info
├── data_collection/         # Data collection module
│   ├── __init__.py
│   └── kipris_collector.py  # KIPRIS patent data collector
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── database.py          # Database connection settings (gitignored)
│   └── settings.py          # Project settings - API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

### KIPRIS Patent Data Collection

```python
from data_collection.kipris_collector import KIPRISCollector
from config.settings import KIPRIS_API_KEY

# Create collector
collector = KIPRISCollector(api_key=KIPRIS_API_KEY)

# Collect and save to JSON file
# limit=None: collect all (until API rate limit)
collector.collect_and_save(limit=None)
```

## Collected Data Structure

### Patent Data (`data/patent/kipris_data.json`)

Each record contains:
- `tech_aplct_id`: Patent application number
- `inpt_mbr_id`: Professor employee ID
- `kipris_index_no`: KIPRIS index number
- `kipris_register_status`: Registration status
- `kipris_application_date`: Application date
- `kipris_abstract`: Patent abstract
- `kipris_application_name`: Invention title
- `professor_info`: Professor information (all fields from `v_emp1` table)
  - `EMP_NO`: Employee number
  - `NM`: Name
  - `HG_NM`: Department name
  - And other fields from `v_emp1` table

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
   - Copy `config/database.py.example` to `config/database.py` and set database credentials
   - Copy `config/settings.py.example` to `config/settings.py` and set KIPRIS API key

## Development Environment

- Python 3.11
- See `requirements.txt` for required packages

## License

This project is for internal research purposes.
