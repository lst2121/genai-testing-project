import pandas as pd
import pytest

df = pd.read_csv("tests/data/ragas_output.csv")

def assert_ragas_metric_above(metric_series, metric_name: str, threshold: float):
    for idx, value in enumerate(metric_series):
        assert value >= threshold, (
            f"{metric_name} too low at index {idx}: {value:.2f} < {threshold} "
            f"→ Q: {df['user_input'][idx]} | A: {df['response'][idx]}"
        )

def test_faithfulness_above_0_5():
    assert_ragas_metric_above(df["faithfulness"], "faithfulness", 0.5)

def test_answer_relevancy_above_0_7():
    assert_ragas_metric_above(df["answer_relevancy"], "answer_relevancy", 0.7)

def test_context_precision_above_0_6():
    assert_ragas_metric_above(df["context_precision"], "context_precision", 0.6)

def test_context_recall_above_0_6():
    assert_ragas_metric_above(df["context_recall"], "context_recall", 0.6)
