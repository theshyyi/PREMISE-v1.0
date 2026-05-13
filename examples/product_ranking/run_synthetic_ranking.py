"""Synthetic product-ranking example."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from premise.product_ranking import average_rank, topsis_rank, consensus_rank


def main() -> None:
    df = pd.DataFrame(
        {
            "product": ["Product_A", "Product_B", "Product_C"],
            "RMSE": [1.20, 0.95, 1.05],
            "KGE": [0.72, 0.69, 0.78],
            "POD": [0.80, 0.76, 0.83],
            "FAR": [0.18, 0.12, 0.20],
        }
    )
    directions = {"RMSE": "cost", "KGE": "benefit", "POD": "benefit", "FAR": "cost"}

    avg = average_rank(df, "product", directions)
    topsis = topsis_rank(df, "product", directions)
    consensus = consensus_rank({"average": avg, "topsis": topsis}, "product")

    print("Average-rank result:")
    print(avg.to_string(index=False))
    print("\nConsensus result:")
    print(consensus[["product", "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
