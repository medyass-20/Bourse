import pandas as pd

def calculate_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("Returns matrix cannot be empty.")

    if len(returns) < 2:
        raise ValueError(
            "At least two return observations are required "
            "to calculate covariance."
        )

    covariance_matrix = returns.cov()

    if covariance_matrix.isna().any().any():
        raise ValueError("Covariance matrix contains NaN values.")

    return covariance_matrix
