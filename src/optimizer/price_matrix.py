import pandas as pd

def prepare_price_matrix(
    prices: pd.DataFrame,

    selected_tickers: list[str],

    prediction_date: str | pd.Timestamp,

    lookback_sessions: int = 252,

    price_column: str = "close"

) -> pd.DataFrame:
    """
    Prepare an aligned historical price matrix
    from an already-validated prices DataFrame.
    """

    if not selected_tickers:
        raise ValueError("selected_tickers cannot be empty.")

    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive.")

    if price_column not in prices.columns:
        raise ValueError(
            f"Price column '{price_column}' not found."
        )

    prediction_date = pd.to_datetime(prediction_date)

    # Filter selected stocks and historical dates
    historical_prices = prices.loc[
        (prices["ticker"].isin(selected_tickers))
        & (prices["date"] <= prediction_date),
        ["date", "ticker", price_column]
    ]

    # Convert long format to wide format
    price_matrix = historical_prices.pivot(
        index="date",
        columns="ticker",
        values=price_column
    )

    # Preserve selected ticker order
    price_matrix = price_matrix.reindex(
        columns=selected_tickers
    )

    # Sort chronologically
    price_matrix = price_matrix.sort_index()

    # Keep the required historical window
    required_observations = lookback_sessions + 1

    price_matrix = price_matrix.tail(
        required_observations
    )

    # Check sufficient history
    if len(price_matrix) < required_observations:
        raise ValueError(
            f"Need {required_observations} price observations, "
            f"but found {len(price_matrix)}."
        )

    # Check completeness of the selected window
    if price_matrix.isna().any().any():
        raise ValueError(
            "Incomplete historical price matrix. "
            "Some selected stocks have missing observations."
        )

    return price_matrix

