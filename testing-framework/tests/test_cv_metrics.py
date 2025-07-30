"""
PyTest: cross‑validation QA guard‑rails + visualisation

Usage
-----
$ pytest -q                               # run tests
# After a successful run you will find `cv_metrics_chart.png`
"""

import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── Configurable QA thresholds ────────────────────────────────────────────────
THRESHOLDS = {
    "accuracy": 0.95,          # min acceptable mean score
    "precision_macro": 0.95,
    "recall_macro": 0.95,
    "f1_macro": 0.95,
}
MAX_STD = 0.05                # max allowed std‑dev across folds
N_FOLDS = 5                   # k in k‑fold CV
PLOT_PATH = Path("cv_metrics_chart.png")

# ── Helper: perform k‑fold CV and return fold scores for each metric ─────────
def run_cv():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier()
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    results = {m: cross_val_score(model, X, y, cv=N_FOLDS, scoring=m) for m in metrics}
    return results

# ── PyTest test case ─────────────────────────────────────────────────────────
def test_cv_metrics_pass_thresholds():
    scores = run_cv()

    # 1) Numerical assertions
    for metric, folds in scores.items():
        mean_score = folds.mean()
        std_score = folds.std()
        assert mean_score >= THRESHOLDS[metric], (
            f"{metric} mean {mean_score:.4f} below threshold {THRESHOLDS[metric]}"
        )
        assert std_score <= MAX_STD, (
            f"{metric} std {std_score:.4f} exceeds max allowed {MAX_STD}"
        )

    # 2) Plot & save chart only if assertions pass
    folds_range = range(1, N_FOLDS + 1)
    bar_width = 0.2
    plt.figure(figsize=(10, 6))

    for i, (metric, folds) in enumerate(scores.items()):
        # Shift bars sideways per metric
        positions = [f + i * bar_width for f in folds_range]
        plt.bar(positions, folds, width=bar_width, label=metric.replace("_macro", "").title())

    plt.xlabel("Fold Number")
    plt.ylabel("Score")
    plt.title("Cross‑Validation Scores Across Folds")
    plt.xticks(
        [f + 1.5 * bar_width for f in folds_range],
        [f"Fold {i}" for i in folds_range],
    )
    plt.ylim(0.8, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)      # <-- chart written to file
    plt.close()

    # Optional: prove file exists
    assert PLOT_PATH.exists(), "Chart file was not saved"

