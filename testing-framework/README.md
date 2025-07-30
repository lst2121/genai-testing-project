# Testing Framework

A comprehensive testing framework for RAG evaluation, cross-validation, and quality assurance.

## Project Structure

### Core Test Files
- **`test_ragas_metrics.py`** - RAGAS evaluation metric assertions and validation
- **`test_cv_metrics.py`** - Cross-validation testing with performance visualization
- **`test_slot_metrics.py`** - Slot classification testing and token-level analysis
- **`test_openai_chat_schema.py`** - Schema validation for OpenAI chat completions
- **`test_ragas_quality.py`** - Quality threshold testing and gate enforcement

### Source Code
- **`source/my_model_output.py`** - Model output handling and processing utilities

### Test Configuration
- **`conftest.py`** - Pytest configuration and shared test fixtures

### Helper Modules
- **`helpers/ragas_assertions.py`** - Custom assertion functions for RAGAS metrics

### RAGAS Evaluation Tools
- **`ragas_eval/ragas_eval_script.py`** - Main RAGAS evaluation execution script
- **`ragas_eval/debug_analysis.py`** - Debugging and analysis utilities for RAGAS results
- **`ragas_eval/convert_testset_to_ragas.py`** - Test set conversion utilities

## Testing Capabilities

### RAG Evaluation Metrics
- Faithfulness (hallucination detection)
- Context Recall (completeness of retrieval)
- Context Precision (relevance of documents)
- Answer Relevancy (response quality)

### Cross-Validation Testing
- K-fold cross-validation for model performance assessment
- Automated threshold checking and validation
- Performance visualization and chart generation

### Slot Classification
- Token-level slot detection metrics
- Precision, recall, and F1 scoring
- Threshold-based quality gates

### Schema Validation
- OpenAI chat completion schema validation
- Response format verification and compliance checking

## Usage

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_ragas_metrics.py
pytest tests/test_cv_metrics.py
pytest tests/test_slot_metrics.py

# Run RAGAS evaluation
python tests/ragas_eval/ragas_eval_script.py

# Generate cross-validation charts
pytest tests/test_cv_metrics.py -v
```

## Quality Gates

The framework enforces the following quality thresholds:
- Faithfulness > 0.7
- Context Recall > 0.7
- Context Precision > 0.6
- Answer Relevancy > 0.7
- Slot Recall > 0.8

## Output Files

- **`cv_metrics_chart.png`** - Generated cross-validation performance visualization 