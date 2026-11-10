from pathlib import Path
import pandas as pd

def save_allocations(
    allocations: pd.DataFrame,

    rebalance_date: pd.Timestamp,

    output_path: str | Path,

    strategy: str = "minimum_variance"
) -> None:

    if allocations.empty:
        raise ValueError("Allocations cannot be empty.")

    required_columns = {"ticker", "weight", "amount_usd"}

    if not required_columns.issubset(allocations.columns):
        raise ValueError(
            f"Allocations must contain: {required_columns}"
        )

    output = allocations.copy()

    output["rebalance_date"] = pd.to_datetime(rebalance_date)
    output["strategy"] = strategy

    output = output[
        [
            "rebalance_date",
            "ticker",
            "weight",
            "amount_usd",
            "strategy"
        ]
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output.to_parquet(
        output_file,
        engine="pyarrow",
        index=False
    )

    print(f"Allocations saved successfully: {output_file}")
