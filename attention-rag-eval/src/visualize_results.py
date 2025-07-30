import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


RAGAS_CSV = Path("data/ragas_outputs/ragas_results.csv")


def plot_metric_histograms(df: pd.DataFrame):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.histplot(df[metric], ax=axes[i], kde=True, bins=10, color="skyblue")
        axes[i].set_title(f"{metric.title()} Distribution")
        axes[i].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("data/ragas_outputs/metric_histograms.png")
    plt.show()


def plot_metric_correlation(df: pd.DataFrame):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    plt.figure(figsize=(8, 6))
    corr = df[metrics].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("RAGAS Metric Correlation Matrix")
    plt.savefig("data/ragas_outputs/metric_correlation.png")
    plt.show()


def plot_low_faithfulness_examples(df: pd.DataFrame, threshold: float = 0.6, top_n: int = 5):
    low = df[df["faithfulness"] < threshold].sort_values("faithfulness").head(top_n)
    for _, row in low.iterrows():
        print("\n---\n")
        print(f"Q: {row['user_input']}")
        print(f"A: {row['response']}")
        print(f"Faithfulness: {row['faithfulness']:.2f} | Answer Rel: {row['answer_relevancy']:.2f}")
        print(f"Context Precision: {row['context_precision']:.2f} | Recall: {row['context_recall']:.2f}")


def main():
    if not RAGAS_CSV.exists():
        raise FileNotFoundError(f"RAGAS results not found at {RAGAS_CSV}")

    df = pd.read_csv(RAGAS_CSV)

    plot_metric_histograms(df)
    plot_metric_correlation(df)
    plot_low_faithfulness_examples(df)


if __name__ == "__main__":
    main()
