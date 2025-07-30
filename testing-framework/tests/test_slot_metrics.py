import pytest
from source.my_model_output import true_slots, pred_slots  # ✅ correct import path

def token_level_slot_metrics(y_true, y_pred):
    tp = fp = fn = 0
    for true_seq, pred_seq in zip(y_true, y_pred):
        for t, p in zip(true_seq, pred_seq):
            if t == p and t != "O":
                tp += 1
            elif t != p:
                if p != "O":
                    fp += 1
                if t != "O":
                    fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

@pytest.mark.parametrize("threshold", [0.65])
def test_slot_recall_above_threshold(threshold):
    _, recall, _ = token_level_slot_metrics(true_slots, pred_slots)
    assert recall >= threshold, f"Slot recall too low: {recall:.2f} < {threshold}"
