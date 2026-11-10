import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio( covariance_matrix: pd.DataFrame, max_weight: float = 0.60 ) -> pd.Series:

    # Number of selected stocks
    n_assets = len(covariance_matrix.columns)

    if n_assets == 0:
        raise ValueError("Covariance matrix cannot be empty.")

    if covariance_matrix.shape != (n_assets, n_assets):
        raise ValueError("Covariance matrix must be square.")

    if len(covariance_matrix) < 1:
        raise ValueError("Covariance matrix must contain assets.")

    if not np.isfinite(covariance_matrix.to_numpy()).all():
        raise ValueError("Covariance matrix contains invalid values.")

    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be greater than 0 and at most 1.")

    if n_assets * max_weight < 1:
        raise ValueError(
            "The maximum weight is too restrictive for the number of assets."
        )

    # Convert covariance matrix to a NumPy array
    covariance = covariance_matrix.to_numpy(dtype=float)

    # Objective: minimize portfolio variance
    def portfolio_variance(weights):
        return weights.T @ covariance @ weights

    # Constraint: sum of weights must equal 1
    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    # Bounds: each weight must be between 0 and max_weight
    bounds = [(0, max_weight) for _ in range(n_assets)]

    # Initial allocation: equal weights
    initial_weights = np.full(n_assets, 1 / n_assets)

    # Run optimization
    result = minimize(
        fun=portfolio_variance,

        x0=initial_weights,

        method="SLSQP",

        bounds=bounds,

        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f"Portfolio optimization failed: {result.message}"
        )

    # Associate each weight with its stock ticker
    weights = pd.Series(
        result.x,
        index=covariance_matrix.columns,
        name="weight"
    )

    return weights
