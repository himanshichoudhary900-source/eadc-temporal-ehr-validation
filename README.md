# Temporal EHR Validation System

Multi-Agent Framework for Explainable Temporal Consistency Validation in Electronic Health Records

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate.bat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add API key to `.env`:
```
GEMINI_API_KEY=AIzaSyD0593D8fsypt9SAvkUYrJmSRkBBk38r5A
```

4. Run test:
```bash
python test.py
```

## Structure

- `data/` - Synthetic EHR timelines
- `agents/` - 4 validation agents
- `core/` - Consensus & explanation engine
- `ui/` - Streamlit interface
- `tests/` - Unit tests