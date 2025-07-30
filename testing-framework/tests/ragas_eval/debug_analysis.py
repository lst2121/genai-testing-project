import pandas as pd
import matplotlib.pyplot as plt

def analyze_ragas_results(df):
    """
    Analyze RAGAS results and provide debugging information.
    
    Args:
        df: DataFrame with RAGAS evaluation results
    """
    print("=== RAGAS Results Analysis ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n=== Sample Data ===")
    print(df.head())
    
    print("\n=== Column Types ===")
    print(df.dtypes)
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Check if faithfulness column exists
    if 'faithfulness' in df.columns:
        print(f"\n=== Faithfulness Statistics ===")
        print(f"Min faithfulness: {df['faithfulness'].min()}")
        print(f"Max faithfulness: {df['faithfulness'].max()}")
        print(f"Mean faithfulness: {df['faithfulness'].mean()}")
        print(f"Number of scores < 0.5: {(df['faithfulness'] < 0.5).sum()}")
        print(f"Number of scores >= 0.5: {(df['faithfulness'] >= 0.5).sum()}")
        
        # Show all faithfulness scores
        print(f"\n=== All Faithfulness Scores ===")
        for idx, score in enumerate(df['faithfulness']):
            print(f"Row {idx}: {score}")
    else:
        print("\n'faithfulness' column not found!")
        print("Available columns:", list(df.columns))

def plot_ragas_results(df):
    """
    Create plots for RAGAS results with error handling.
    
    Args:
        df: DataFrame with RAGAS evaluation results
    """
    if 'faithfulness' not in df.columns:
        print("'faithfulness' column not found!")
        return
    
    # Check if there are any low scores
    low_scores = df[df["faithfulness"] < 0.5]
    high_scores = df[df["faithfulness"] >= 0.5]
    
    print(f"Low faithfulness scores (< 0.5): {len(low_scores)}")
    print(f"High faithfulness scores (>= 0.5): {len(high_scores)}")
    
    if len(low_scores) > 0:
        plt.figure(figsize=(10, 5))
        low_scores[["user_input", "faithfulness"]].plot.barh(
            x="user_input", y="faithfulness", legend=False
        )
        plt.title("Low Faithfulness QnAs")
        plt.xlabel("Faithfulness")
        plt.tight_layout()
        plt.show()
    else:
        print("All faithfulness scores are >= 0.5!")
        
        # Plot all scores instead
        plt.figure(figsize=(10, 5))
        df[["user_input", "faithfulness"]].plot.barh(
            x="user_input", y="faithfulness", legend=False
        )
        plt.title("All Faithfulness Scores")
        plt.xlabel("Faithfulness")
        plt.tight_layout()
        plt.show()

# Example usage:
# analyze_ragas_results(df)
# plot_ragas_results(df) 