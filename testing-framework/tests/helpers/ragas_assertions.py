# tests/helpers/ragas_assertions.py

from ragas import evaluate
import pandas as pd

def assert_ragas_metric_above(
    dataset,
    metric,
    threshold: float,
    metric_name: str = None
):
    """
    Evaluate a RAGAS metric and assert it meets the threshold for all QnA rows.

    Parameters:
        dataset (Dataset): HuggingFace dataset in RAGAS format.
        metric (ragas Metric): e.g., faithfulness, context_precision
        threshold (float): Minimum acceptable value.
        metric_name (str): Optional override for name in error message.
    """
    results = evaluate(dataset=dataset, metrics=[metric])
    df = results.to_pandas()
    
    # Find the metric column (it should be the last column)
    metric_columns = [col for col in df.columns if col not in ['user_input', 'retrieved_contexts', 'response', 'reference']]
    name = metric_name or metric_columns[0] if metric_columns else df.columns[-1]

    for i, score in enumerate(df[name]):
        assert score >= threshold, f"[{name}] QnA #{i} failed: {score:.2f} < {threshold:.2f}"
