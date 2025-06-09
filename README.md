# Article Analysis Agent

This project implements a LangGraph-based agent that analyzes articles and provides improvement suggestions while comparing results with OpenAI and Perplexity.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

Run the FastAPI server:
```bash
uvicorn main:app --reload
```

Send a POST request to `/analyze` with your article text to get analysis and suggestions. 