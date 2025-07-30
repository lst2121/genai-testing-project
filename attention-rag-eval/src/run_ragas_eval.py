import ast
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()


def load_predictions(csv_path: str | Path):
    df = pd.read_csv(csv_path)

    # Convert stringified list to actual list
    if df["retrieved_chunks"].dtype == object:
        df["retrieved_chunks"] = df["retrieved_chunks"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df


def convert_to_ragas_records(df: pd.DataFrame):
    return [
        {
            "question": row["question"],
            "contexts": row["retrieved_chunks"],
            "answer": row["response"],
            "ground_truth": row["ground_truth"],
        }
        for _, row in df.iterrows()
    ]


def main():
    pred_csv = Path("data/ragas_outputs/rag_predictions.csv")
    if not pred_csv.exists():
        raise SystemExit(f"Predictions CSV not found at {pred_csv}")

    print("Loading predictions …")
    df_pred = load_predictions(pred_csv)
    records = convert_to_ragas_records(df_pred)

    print("Running RAGAS evaluation …")
    ragas_ds = Dataset.from_list(records)
    results = evaluate(
        dataset=ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    ).to_pandas()

    out_path = Path("data/ragas_outputs/ragas_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print("RAGAS results saved to", out_path.resolve())

        # Save metrics-only CSV
    metrics_only = results[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
    metrics_path = Path("data/ragas_outputs/ragas_metrics_only.csv")
    metrics_only.to_csv(metrics_path, index=False)
    print("RAGAS metrics-only file saved to", metrics_path.resolve())

    # Print quick averages
    print("\nAverage Metrics:")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results[m].mean():.3f}")


if __name__ == "__main__":
    main()
