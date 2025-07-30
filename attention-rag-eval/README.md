# Attention RAG Evaluation

RAG pipeline evaluation using the "Attention Is All You Need" paper.

## Overview

Complete pipeline for:
- Document ingestion and chunking
- Vector database setup with ChromaDB
- Test set generation using RAGAS
- RAG pipeline implementation
- Evaluation with RAGAS metrics
- Results visualization

## Usage

```bash
# Ingest and chunk the paper
python src/ingest_paper.py

# Generate test questions
python src/generate_testset.py

# Run RAG pipeline
python src/run_rag_pipeline_on_testset.py

# Evaluate with RAGAS
python src/run_ragas_eval.py

# Visualize results
python src/visualize_results.py
```

## Evaluation Metrics

- **Faithfulness**: Answer consistency with retrieved context
- **Answer Relevancy**: Answer relevance to question
- **Context Precision**: Precision of retrieved context
- **Context Recall**: Recall of relevant context

## Output Files

- `data/ragas_outputs/generated_testset.csv` - Test questions
- `data/ragas_outputs/rag_predictions.csv` - RAG predictions
- `data/ragas_outputs/ragas_results.csv` - Evaluation results
- `data/ragas_outputs/metric_histograms.png` - Metric distributions
- `data/ragas_outputs/metric_correlation.png` - Metric correlations 