import pytest
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from tests.helpers.ragas_assertions import assert_ragas_metric_above

# Create a simple test dataset
from datasets import Dataset
ragas_ds = Dataset.from_dict({
    "question": ["What is RAG?"],
    "contexts": [["RAG is Retrieval-Augmented Generation"]],
    "answer": ["RAG is a technique that combines retrieval with generation"],
    "ground_truth": ["RAG is Retrieval-Augmented Generation"]
})  

@pytest.mark.parametrize("metric,threshold", [
    (faithfulness, 0.70),
    (answer_relevancy, 0.75),
    (context_precision, 0.60),
    (context_recall, 0.60),
])
def test_ragas_quality(metric, threshold):
    assert_ragas_metric_above(ragas_ds, metric, threshold)
