import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Sample data
data = [
    {
        "question": "Who was the man behind The Chipmunks?",
        "answer": "David Seville",
        "contexts": [
            "Alvin and the Chipmunks were created by Ross Bagdasarian Sr., who performed under the stage name David Seville."
        ],
        "ground_truth": "David Seville"
    },
    {
        "question": "What claimed the life of singer Kathleen Ferrier?",
        "answer": "Cancer",
        "contexts": [
            "Kathleen Ferrier died of cancer in 1953 at the age of 41. She was one of Britain's most celebrated contraltos."
        ],
        "ground_truth": "Cancer"
    },
    {
        "question": "What is the Japanese share index called?",
        "answer": "Nikkei",
        "contexts": [
            "The Nikkei 225, commonly called the Nikkei, is Japan's premier stock market index, tracking the performance of 225 large companies on the Tokyo Stock Exchange."
        ],
        "ground_truth": "Nikkei"
    },
    {
        "question": "What was the name of Michael Jackson's autobiography?",
        "answer": "Moonwalk",
        "contexts": [
            "Michael Jackson's autobiography, 'Moonwalk', was published in 1988 and provides insights into his personal life and career."
        ],
        "ground_truth": "Moonwalk"
    },
    {
        "question": "In what year's Olympics were electric timing devices first used?",
        "answer": "1912",
        "contexts": [
            "Electric timing devices were first introduced at the 1912 Olympic Games in Stockholm, Sweden, to ensure accurate race results."
        ],
        "ground_truth": "1912"
    }
]


dataset = Dataset.from_list(data)

results = evaluate(
    dataset=dataset,
    metrics=[answer_relevancy, faithfulness, context_precision, context_recall]
)

# Save to CSV
df = results.to_pandas()
df.to_csv("data/ragas_output.csv", index=False)
print("RAGAS results saved to data/ragas_output.csv")
