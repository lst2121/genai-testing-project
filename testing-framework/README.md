# Testing Framework

Testing framework for RAG evaluation, cross-validation, and quality assurance.

## Test Files

- **`test_ragas_metrics.py`** - RAGAS evaluation metric assertions
- **`test_cv_metrics.py`** - Cross-validation testing with visualization
- **`test_slot_metrics.py`** - Slot classification testing
- **`test_openai_chat_schema.py`** - Schema validation testing
- **`test_ragas_quality.py`** - Quality threshold testing

## What's Tested

### RAG Evaluation Metrics
- Faithfulness (hallucination detection)
- Context Recall (completeness of retrieval)
- Context Precision (relevance of documents)
- Answer Relevancy (response quality)

### Cross-Validation Testing
- K-fold cross-validation for model performance
- Automated threshold checking
- Performance visualization

### Slot Classification
- Token-level slot detection metrics
- Precision, recall, and F1 scoring
- Threshold-based quality gates

### Schema Validation
- OpenAI chat completion schema validation
- Response format verification

## Usage

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_ragas_metrics.py
pytest tests/test_cv_metrics.py
pytest tests/test_slot_metrics.py
```

## Quality Gates

- Faithfulness > 0.7
- Context Recall > 0.7
- Context Precision > 0.6
- Answer Relevancy > 0.7
- Slot Recall > 0.8 