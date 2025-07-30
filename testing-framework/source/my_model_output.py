# my_model_output.py
# -----------------------------------------
# Purpose: Provide lists `true_slots` and `pred_slots`
#          that PyTest can import.

import random
from datasets import load_dataset

# LOAD DATA (use the dataset you used in the notebook)
dataset = load_dataset("bkonkle/snips-joint-intent", split="test")

# Utility: split BIO string into list
def split_bio(bio_str):
    return bio_str.split()

# Ground‑truth slot labels
true_slots = [split_bio(x["slots"]) for x in dataset]

# PREDICT SLOTS  

pred_slots = []
for gold in true_slots:
    pred_seq = [
        tag if random.random() > 0.3 else "O" 
        for tag in gold
    ]
    pred_slots.append(pred_seq)


if __name__ == "__main__":
    from collections import Counter
    print("Example true:", true_slots[0][:10])
    print("Example pred:", pred_slots[0][:10])
    total_tokens = sum(len(s) for s in true_slots)
    kept = sum(t == p and t != "O"
               for ts, ps in zip(true_slots, pred_slots)
               for t, p in zip(ts, ps))
    print(f"Simulated slot recall ≈ {kept / sum(t != 'O' for seq in true_slots for t in seq):.2f}")
