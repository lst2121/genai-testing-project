# RAG Evaluation Framework

A framework for building and evaluating Retrieval-Augmented Generation (RAG) systems.

## Project Structure

```
├── attention-rag-eval/          # Complete RAG pipeline
├── notebooks-demo/              # Jupyter notebooks
├── testing-framework/           # Testing suite
└── README.md
```

## Setup

1. **Install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Add API keys to `.env`:**
```env
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here  # Optional - will fallback to OpenAI
```

3. **Add the paper (for attention-rag-eval):**
Place "Attention Is All You Need.pdf" in `attention-rag-eval/data/paper/`

## Getting Started

### RAG Pipeline
```bash
cd attention-rag-eval
python src/ingest_paper.py
python src/generate_testset.py
python src/run_rag_pipeline_on_testset.py
python src/run_ragas_eval.py
```

### Testing
```bash
cd testing-framework
pytest tests/
```

### Notebooks
```bash
cd notebooks-demo
jupyter notebook
```

## What's Included

### attention-rag-eval/
- Complete RAG system with document ingestion
- Automated test generation using RAGAS
- Evaluation metrics and quality assessment
- Applied to "Attention Is All You Need" paper

### testing-framework/
- Automated test suites with PyTest
- RAGAS metric validation
- Cross-validation testing
- Quality thresholds and gates

### notebooks-demo/
- Interactive examples of test generation
- RAGAS evaluation workflows
- Practical demonstrations
