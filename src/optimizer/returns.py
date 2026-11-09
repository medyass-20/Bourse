import pandas as pd

def calculate_returns( price_matrix: pd.DataFrame ) -> pd.DataFrame:
    """
    Calculate simple daily returns for each selected stock.

    Parameters
    ----------
    price_matrix : pd.DataFrame
        Historical prices with dates as index
        and tickers as columns.

    Returns
    -------
    pd.DataFrame
        Daily returns for each stock.
    """

    if price_matrix.empty:
        raise ValueError("Price matrix cannot be empty.")

    # Calculate simple daily returns
    returns = price_matrix.pct_change(fill_method=None)

    # Remove the first row containing NaN
    returns = returns.dropna()

    return returns
