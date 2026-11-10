import numpy as np
import pandas as pd

def calculate_allocation_amounts( weights: pd.Series, total_capital: float = 10_000.00 ) -> pd.DataFrame:

    if weights.empty:
        raise ValueError("Weights cannot be empty.")

    if not np.isfinite(total_capital) or total_capital <= 0:
        raise ValueError("Total capital must be a positive finite number.")

    weight_values = weights.to_numpy(dtype=float)

    if not np.isfinite(weight_values).all():
        raise ValueError("Weights contain invalid values.")

    if (weight_values < 0).any():
        raise ValueError("Weights cannot be negative.")

    if not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    # Calculate allocation amounts
    amounts = (weight_values * total_capital).round(2)

    # Correct any rounding difference so the total matches the capital
    rounding_difference = round(total_capital - amounts.sum(), 2)
    largest_allocation_index = int(np.argmax(weight_values))
    amounts[largest_allocation_index] += rounding_difference

    # Build the allocation DataFrame
    allocations = pd.DataFrame({
        "ticker": weights.index,
        "weight": weight_values,
        "amount_usd": amounts
    })

    return allocations
